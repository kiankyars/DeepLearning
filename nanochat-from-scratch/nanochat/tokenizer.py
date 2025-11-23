"""
SERIES PRINCIPLES
==========================

I'm doing this with you so you can learn through my own problem-solving process.
I will write pseudocode for each function before implementing it.
I demonstrate everything through small examples.
I'm going to do this in one take.

"""
import regex as re
import pickle
import os
import copy

def get_pairs(ids, counts):
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, index):
    '''
    this function will iterate over ids and every time
    it sees a instance of pair, it will take that pair
    and instead put index, then it will return the list
    list = [1,2,3,4,1,2,3]
    merge(list, (1,2), 257)
    list = [257,3,4,257,3]
    '''
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(index)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class RegexTokenizer:
    def __init__(self):
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

    def get_pattern(self):
        return self.pattern

    def train_from_iterator(self, text_iterator, vocab_size):
        """Train from streaming iterator without loading all text into memory"""
        from collections import Counter
        # Step 1: Count unique chunks across all text (streaming)
        chunk_counts = Counter()
        for text in text_iterator:
            chunks = re.findall(self.pattern, text)
            chunk_counts.update(chunks)
        # Step 2: Encode unique chunks once
        # Counter({' The': 1, ' response': 1, ' in': 1, ' Berlin': 1, ' was': 1, ' panic': 1, '.': 1})
        unique_chunks = list(chunk_counts.keys())
        # [' The', ' response', ' in', ' Berlin', ' was', ' panic', '.']
        encoded_chunks = [list(chunk.encode('utf-8')) for chunk in unique_chunks]
        # [[32, 84, 104, 101], [32, 114, 101, 115, 112, 111, 110, 115, 101], [32, 105, 110], [32, 66, 101, 114, 108, 105, 110], [32, 119, 97, 115], [32, 112, 97, 110, 105, 99], [46]]
        counts = [chunk_counts[chunk] for chunk in unique_chunks]
        # The above gives the number of times that each unique chunk appears.
        
        # Step 3: Train on unique chunks with their counts
        number_merges = vocab_size - 256
        for i in range(number_merges):
            if i % 100 == 0:
                print(f"Merge {i}/{number_merges}")
            
            # Count pairs weighted by chunk frequency
            pairs = {}
            for encoded_chunk, count in zip(encoded_chunks, counts):
                for j in range(len(encoded_chunk) - 1):
                    pair = (encoded_chunk[j], encoded_chunk[j+1])
                    pairs[pair] = pairs.get(pair, 0) + count
            
            if not pairs:
                break
            
            pair = max(pairs, key=pairs.get)
            index = 256 + i
            
            # Merge in all unique chunks
            for j in range(len(encoded_chunks)):
                encoded_chunks[j] = merge(encoded_chunks[j], pair, index)
            
            self.merges[pair] = index
        
    def encode(self, text):
        text_chunks = re.findall(self.pattern, text)
        encoded_text = []
        for text_chunk in text_chunks:
            encoded_chunk = self._encode_chunk(text_chunk)
            encoded_text.extend(encoded_chunk)
        return encoded_text
    
    def _encode_chunk(self, text):
        '''
        self.merges is important here
        
        we get text, and then we convert that text to byte strings, then to integers
        and then we iterate over the text until all pairs of merges that are
        *** we merge the pairs in the order they were merged at training ***
        possible under the trained tokenizer have been completed
        '''
        ids = list(text.encode('utf-8'))
        for pair, index in self.merges.items():
            ids = merge(ids, pair, index)
        return ids


    def decode(self, ids):
        '''
        decode gets ids
        1. convert the ids to their byte strings
        2. convert the byte strings to strings via the vocabulary
        3. then return the decoded_text
        .decode('utf-8')
        [239, 256]
        [b'xa', b'sa']
        b'xasa'
        output

        '''
        byte_strings = b''.join([bytes(self.vocabulary[i]) for i in ids])
        decoded_text = byte_strings.decode('utf-8')
        return decoded_text

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "merges":self.merges,
                "vocabulary":self.vocabulary,
                "pattern":self.pattern.pattern
            },
            f)

    @classmethod
    def load(cls, path):
        tokenizer = cls(300, r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        with open(path, "rb") as f:
            data = pickle.load(f)
            tokenizer.merges = data["merges"]
            tokenizer.vocabulary = data["vocabulary"]
            tokenizer.pattern = data["pattern"]
        return tokenizer

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

import tiktoken
from functools import lru_cache

def get_mergeable_ranks(tokenizer):
    # add base 256 strings
    mergeable_ranks = {}
    for i in range(256):
        mergeable_ranks[bytes([i])] = i
    # add merges tokens in the order they were made and with the bytes, not the int pairs, as is the case in self.merges
    token_bytes = {i: bytes([i]) for i in range(256)}
    for (pair, merged_index) in tokenizer.merges.items():
        merges_bytes = token_bytes[pair[0]] + token_bytes[pair[1]]
        token_bytes[merged_index] = merges_bytes
        mergeable_ranks[merges_bytes] = merged_index
    return mergeable_ranks


class NanoChatTokenizer:
    def __init__(self, enc, bos_token) -> None:
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    def get_bos_token_id(self):
        return self.bos_token_id

    
    def __repr__(self) -> str:
        return self.enc.name

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # Step 1 train python tokenizer use regextokenizer, then throw it out!
        # self.merges() is all we want and need from the tokenizer training
        # Step 2 convert self.merges to mergeable_ranks
        # self. merges is a mapping from pairs to indices, 
        # but mergeable_ranks is the opposite
        # step 3. add special tokens
        # step4 create and return the tiktoken.encoding object
        tokenizer = RegexTokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special)
        mergable_ranks = get_mergeable_ranks(tokenizer)
        tokens_offset = len(mergable_ranks)
        special_tokens = {
            token: tokens_offset + i
            for i, token in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="nanochat",
            pat_str=tokenizer.get_pattern().pattern,
            mergeable_ranks=mergable_ranks,
            special_tokens=special_tokens
        )
        return cls(enc, "<|bos|>")

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads)
            if prepend is not None:
                for i in ids:
                    i.insert(0, prepend_id)
            if append is not None:
                for j in ids:
                    j.append(append_id)
        else:
            raise ValueError(f"invalid input type: {type(text)}")
        return ids
    def decode(self, ids):
        return self.enc.decode(ids)
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc,f)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")
    
    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    
    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            # check user vs assistant
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                add_tokens(user_start, 0)
                value_ids = self.encode(content)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
                # assitant
            elif message["role"] == "assistant":
                # add assistant start tokens
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                    # then we will go straight to add_tokens for assitant end, unless we have unknown content type
                # these are the more nuanced cases
                elif isinstance(content, list):
                    for part in content:
                        # for element in list
                        value_ids = self.encode(part["text"])
                        # encode each element
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                            # if it was text, we add without any other special tokens
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            # add python special tokens in this case
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            # python output, looks like this is the python output of the python generated by the llm
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                # add assitant end tokens
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conservation):
        conversation = copy.deepcopy(conservation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, mask = self.render_conversation(conversation)
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

from nanochat.common import get_base_dir
def get_tokenizer():
    # Purpose: Loads the tokenizer from the standard location (`~/.cache/nanochat/tokenizer/`). Used throughout the codebase instead of hardcoding paths.
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return NanoChatTokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device):
    import torch
    # Purpose: Loads a precomputed tensor `token_bytes[vocab_size]` where `token_bytes[token_id]` = number of bytes that token represents (0 for special tokens).

    # This normalizes by actual byte length, so you can compare tokenizers with different vocab sizes.

    # key-insight: bytes are the invariant—both tokenizers compress the same original bytes, so normalize by that.
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer/token_bytes.pt")
    assert os.path.exists(tokenizer_dir), "Token bytes not found…"
    with open(tokenizer_dir, "rb") as f:
        return torch.load(f, map_location=device)