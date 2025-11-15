"""
  ## Summary of progression

| Step | Tokenizer Type | Training | Inference | Key Addition |
|------|----------------|----------|-----------|--------------|
| 1 | BasicTokenizer | Python | Python | Byte-level BPE |
| 2 | RegexTokenizer | Python | Python | Regex splitting + save/load |
| 3 | RegexTokenizer + Special | Python | Python | Special token handling |
| 4 | NanoChatTokenizer | Python | Tiktoken (C) | Tiktoken backend + conversation rendering |


Method descriptions (one sentence each)
__init__(self, enc, bos_token): Initialize with a tiktoken.Encoding and BOS token string, storing both and caching the BOS token ID.
train_from_iterator(cls, text_iterator, vocab_size): Class method that trains a tokenizer from a text iterator, converts to tiktoken.Encoding, and returns a new instance.
from_directory(cls, tokenizer_dir): Class method that loads a pickled tiktoken.Encoding from disk and returns a new instance.
from_pretrained(cls, tiktoken_name): Class method that loads a pretrained tiktoken encoding by name (e.g., "cl100k_base") and returns a new instance.
get_vocab_size(self): Returns total vocabulary size (regular tokens + special tokens).
get_special_tokens(self): Returns the set of special token strings.
id_to_token(self, id): Decodes a single token ID to its string representation.
encode_special(self, text): Encodes a special token string to its token ID (cached).
get_bos_token_id(self): Returns the cached BOS token ID integer.
encode(self, text, prepend=None, append=None, num_threads=8): Encodes text (str or list) to token IDs, with optional prepend/append tokens, using tiktoken for efficiency.
decode(self, ids): Decodes a list of token IDs back to a Python string.
save(self, tokenizer_dir): Saves the tiktoken.Encoding object to disk as a pickle file.
render_conversation(self, conversation, max_tokens=2048): Tokenizes a chat conversation dict into token IDs and a training mask (1 for assistant tokens to predict, 0 otherwise).
render_for_completion(self, conversation): Tokenizes a conversation for RL, removing the last assistant message and appending <|assistant_start|> to prime generation.


SERIES PRINCIPLES
==========================

I'm doing this with you so you can learn through my own problem-solving process.
I will write pseudocode for each function before implementing it.
I demonstrate everything through small examples.
I'm going to do this in one take.

Why?
Add register_special_tokens() method
Reserve space for special tokens during training
Modify encode() to handle special tokens in text
Modify decode() to handle special token IDs
Add encode_special() helper method
Demo: encode text with special tokens

f you want to show special tokens, do it at the end of Video 3: "Here's how you could add special tokens to the Python tokenizer, but we don't need to since tiktoken handles it."
This aligns with nanochat’s approach: special tokens are added when creating the tiktoken.Encoding, not during Python training.

Why store BOS token id as a class member (one‑liner for your video)
We store the BOS id because render_conversation must always prepend a single, known document delimiter at runtime; different encodings use different literal tokens ("<|bos|>" vs "<|endoftext|>"), so the tokenizer needs the concrete id to insert quickly and deterministically. In short: it’s the one special token the runtime always needs and must be mapped correctly across encodings.

TL;DR: The parameter handles different token names ("<|bos|>" vs "<|endoftext|>"). Storing the ID is a convenience, not a requirement.

"""

import regex as re
import pickle

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
    def __init__(self, vocab_size, pattern):
        self.vocab_size = vocab_size
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
    def train(self, text, verbose=False):
        # encode the text
        # iterate over text, self.vocab_size - 256 times
        # count all of the pairs in a dictionary
        # choose the pair with the highest frequency
        # merge that pair as a new token
        # add that token to the vocab
        # {256: byte_string}
        # add to self.merges = {byte_string: 256}
        assert self.vocab_size >= 256
        number_merges = self.vocab_size - 256
        
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
tokenizer = RegexTokenizer(300, r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
tokenizer.train(text, True)
# tokenizer.save("/Users/kian/Code/DeepLearning/nanochat/tokenizer.pkl")
tokenizer_load = tokenizer.load("/Users/kian/Code/DeepLearning/nanochat/tokenizer.pkl")

print(tokenizer_load.decode(tokenizer_load.encode('are hello')))

# tokenizer = RegexTokenizer(259, r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
# tokenizer.train(text, True)
# print(tokenizer.encode('are hello'))
# print(list('are hello'.encode('utf-8')))
# print(tokenizer.merges)

# print(tokenizer.decode([239, 188, 181, 239, 189, 142, 239, 189, 137, 239, 189, 131, 239, 189, 143, 239, 189, 132, 239, 189, 133, 33, 32, 263, 164, 263, 157, 263, 152, 263, 146, 263, 158, 263, 147, 263, 148, 258, 189]))