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
from collections import Counter

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_stats(ids_freqs):
    """
    Given a dictionary of {(token_ids): frequency}, compute the frequency
    of every adjacent pair of tokens.
    """
    counts = {}
    for ids, freq in ids_freqs.items():
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + freq
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers `ids`, replace all consecutive occurrences
    of `pair` with the new token `idx`.
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last element AND the pair matches
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# -----------------------------------------------------------------------------
# RegexTokenizer
# -----------------------------------------------------------------------------

class RegexTokenizer:
    def __init__(self):
        # Initialize base vocabulary (0-255)
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        # GPT-4 style regex pattern
        self.pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

    def get_pattern(self):
        return self.pattern

    def train_from_iterator(self, text_iterator, vocab_size):
        """
        Train the BPE tokenizer from an iterator of strings.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        
        # 1. Ingestion: Regex split and count unique words
        print(f"Processing iterator...")
        word_counts = Counter()
        
        for text in text_iterator:
            # Find all chunks matching the pattern
            text_chunks = re.findall(self.pattern, text)
            for chunk in text_chunks:
                # Convert string chunk to UTF-8 bytes, then to a tuple of integers
                # We use tuple so it can be a dictionary key
                chunk_ids = tuple(chunk.encode("utf-8"))
                word_counts[chunk_ids] += 1

        print(f"Unique words identified: {len(word_counts)}")
        print(f"Starting BPE training for {num_merges} merges...")

        # 2. Training Loop
        for i in range(num_merges):
            # a. Compute co-occurrence stats for all pairs
            stats = get_stats(word_counts)
            
            # If no pairs are left, we can't merge anymore
            if not stats:
                break
            
            # b. Find the most frequent pair
            # (lexicographical tie-breaking via pair comparison handled by python max)
            pair = max(stats, key=stats.get)
            
            # c. Mint the new token ID
            idx = 256 + i
            
            # d. Record the merge
            self.merges[pair] = idx
            # Optional: add to vocabulary mapping for decoding later
            # (Reconstruct the bytes for the new token)
            vocab_bytes = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
            self.vocabulary[idx] = vocab_bytes
            
            # e. Update the counts dictionary (Apply merge)
            # We rebuild the dictionary because keys (tuples) change
            new_word_counts = {}
            for ids, freq in word_counts.items():
                # Apply the merge to this specific word
                new_ids = merge(list(ids), pair, idx)
                new_ids = tuple(new_ids)
                new_word_counts[new_ids] = new_word_counts.get(new_ids, 0) + freq
            
            word_counts = new_word_counts
            
            # Simple progress logging
            if (i + 1) % 100 == 0:
                print(f"Merge {i + 1}/{num_merges}: {pair} -> {idx} (freq {stats[pair]})")

    def encode(self, text):
        text_chunks = re.findall(self.pattern, text)
        encoded_text = []
        for text_chunk in text_chunks:
            encoded_chunk = self._encode_chunk(text_chunk)
            encoded_text.extend(encoded_chunk)
        return encoded_text
    
    def _encode_chunk(self, text):
        # Start with raw bytes
        ids = list(text.encode('utf-8'))
        # Apply merges in the order they were learned
        # Note: In an optimized implementation, we would iterate through 
        # the text and apply the best available pair iteratively,
        # but iterating through the fixed merge list is the standard reference implementation.
        while len(ids) >= 2:
            stats = get_stats({tuple(ids): 1})
            # Find the pair with the lowest merge index (earliest merge)
            pair_to_merge = None
            min_merge_idx = float('inf')
            
            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_merge_idx:
                        min_merge_idx = self.merges[pair]
                        pair_to_merge = pair
            
            if pair_to_merge:
                ids = merge(ids, pair_to_merge, min_merge_idx)
            else:
                break
                
        return ids

    def decode(self, ids):
        # Filter out special tokens if they are not in self.vocabulary
        # (Assuming standard BPE behavior, special tokens are handled outside or added to vocab)
        byte_parts = []
        for idx in ids:
            if idx in self.vocabulary:
                byte_parts.append(self.vocabulary[idx])
            else:
                # Fallback for debugging or special tokens not in internal vocab
                byte_parts.append(b"") 
        
        byte_string = b"".join(byte_parts)
        # errors='replace' handles cases where split unicode characters appear at boundaries
        return byte_string.decode('utf-8', errors='replace')

    def save(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "merges": self.merges,
                "vocabulary": self.vocabulary,
                "pattern": self.pattern.pattern
            }, f)

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
            tokenizer.merges = data["merges"]
            tokenizer.vocabulary = data["vocabulary"]
            tokenizer.pattern = re.compile(data["pattern"])
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

def get_mergeable_ranks(tokenizer, verbose=True):
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
        if verbose:
            print(f"merged pair {pair} at index {merged_index}")
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

    def get_mergable_ranks(self):
        return self.enc._mergeable_ranks

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