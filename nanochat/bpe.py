"""
Method descriptions

save(self, tokenizer_dir): Saves the tiktoken.Encoding object to disk as a pickle file.

# In other words load. This is just the way it's called for factory methods.
@classmethod
from_directory(cls, tokenizer_dir): Class method that loads a pickled tiktoken.Encoding from disk and returns a new instance.

get_vocab_size(self): Returns total vocabulary size (regular tokens + special tokens).

get_special_tokens(self): Returns the set of special token strings.

from_pretrained(cls, tiktoken_name): Class method that loads a pretrained tiktoken encoding by name (e.g., "cl100k_base") and returns a new instance.

render_conversation(self, conversation, max_tokens=2048): Tokenizes a chat conversation dict into token IDs and a training mask (1 for assistant tokens to predict, 0 otherwise).

render_for_completion(self, conversation): Tokenizes a conversation for RL, removing the last assistant message and appending <|assistant_start|> to prime generation.


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
    def __init__(self, pattern):
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
    def train(self, text, vocab_size_no_special, verbose=False):
        # encode the text
        # iterate over text, self.vocab_size - 256 times
        # count all of the pairs in a dictionary
        # choose the pair with the highest frequency
        # merge that pair as a new token
        # add that token to the vocab
        # {256: byte_string}
        # add to self.merges = {byte_string: 256}
        number_merges = vocab_size_no_special - 256
        
        text_chunks = re.findall(self.pattern, text)
        encoded_chunks = [list(text_chunk.encode('utf-8')) for text_chunk in text_chunks]
        length_initial = sum([len(encoded_chunk) for encoded_chunk in encoded_chunks])

        for i in range(number_merges):
            pairs = {}
            for encoded_chunk in encoded_chunks:
                get_pairs(encoded_chunk, pairs)
            pair = max(pairs, key=pairs.get)
            index = 256 + i
            encoded_chunks = [merge(encoded_chunk, pair, index) for encoded_chunk in encoded_chunks]
            self.merges[pair] = index
            self.vocabulary[index] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
            # print(sorted([(v,k) for k,v in pairs.items()], reverse=True)[:10])
            # return
        print([(k,v)for k, v in  self.vocabulary.items()][355:])


        if verbose:
            length_final = sum([len(encoded_chunk) for encoded_chunk in encoded_chunks])
            compression = length_initial/length_final
            print(length_initial, length_final)
            print(compression)
        # print(ids)
        
        
        # print(sorted([(v,k) for k,v in pairs.items()], reverse=True)[:10])
        # print(sorted(pairs.items(),reverse=True,key=lambda k: pairs[k])[:10])
        
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


text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

# tokenizer = RegexTokenizer(300, r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
# tokenizer.train(text, True)
# tokenizer.save("/Users/kian/Code/DeepLearning/nanochat/tokenizer.pkl")
# tokenizer_load = tokenizer.load("/Users/kian/Code/DeepLearning/nanochat/tokenizer.pkl")
# print(tokenizer_load.decode(tokenizer_load.encode('are hello')))
# tokenizer = RegexTokenizer(259, r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
# tokenizer.train(text, True)
# print(tokenizer.encode('are hello'))
# print(list('are hello'.encode('utf-8')))
# print(tokenizer.merges)
# print(tokenizer.decode([239, 188, 181, 239, 189, 142, 239, 189, 137, 239, 189, 131, 239, 189, 143, 239, 189, 132, 239, 189, 133, 33, 32, 263, 164, 263, 157, 263, 152, 263, 146, 263, 158, 263, 147, 263, 148, 258, 189]))

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

    
    def __repr__(self) -> str:
        return self.enc.name

    @classmethod
    def train_from_iterator(cls, text, vocab_size, pattern):
        # Step 1 train python tokenizer use regextokenizer, then throw it out!
        # self.merges() is all we want and need from the tokenizer training
        # Step 2 convert self.merges to mergeable_ranks
        # self. merges is a mapping from pairs to indices, 
        # but mergeable_ranks is the opposite
        # step 3. add special tokens
        # step4 create and return the tiktoken.encoding object
        tokenizer = RegexTokenizer(pattern)
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train(text, vocab_size_no_special)
        mergable_ranks = get_mergeable_ranks(tokenizer)
        tokens_offset = len(mergable_ranks)
        special_tokens = {
            token: tokens_offset + i
            for i, token in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="nanochat",
            pat_str=pattern,
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
        pass
        # - Chat format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`

        conversation = {
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }

        # Should produce:
        # list= [<|bos|>, <|user_start|>, "Hello!", <|user_end|>, 
        # <|assistant_start|>, "Hi there!", <|assistant_end|>]
        # With mask: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]

        Model generates: <|python_start|> 'hello'.count('l') <|python_end|>
        System executes the code: use_calculator(expr)
        System forces the result: <|output_start|> 2 <|output_end|>
        Model continues generating after the output

        # Training data:
        {"type": "python", "text": "123 + 456"}      # mask=1 (model predicts this)
        {"type": "python_output", "text": "579"}     # mask=0 (system provides this)

        # At inference:
        # Model predicts: "<|python_start|> 123 + 456 <|python_end|>"
        # System runs it: result = 579
        # System injects: "<|output_start|> 579 <|output_end|>"

text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()
tokenizer1 = NanoChatTokenizer.train_from_iterator(text, 300, r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

tokenizer1.save("./nanochat")

print(tokenizer1.get_vocab_size())

tokenizer = NanoChatTokenizer.from_directory("./nanochat")

print(tokenizer.encode('Hello') == tokenizer('Hello'))

print(tokenizer('Hello I am so happy today'))

print(tokenizer.decode(tokenizer("hello")))


    

gpt2= NanoChatTokenizer.from_pretrained("gpt2")

print(gpt2.get_special_tokens(), gpt2.get_vocab_size())
# print((gpt2.enc.special_tokens_set.pop()))
# print(gpt2.encode_special(gpt2.enc.special_tokens_set.pop()))