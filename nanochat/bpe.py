```python
# ADD: Pattern and regex compilation
import regex as re
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
    def __init__(self, pattern=None):
        self.pattern = re.compile(pattern or GPT4_SPLIT_PATTERN)
        # ... rest of init

# MODIFY: train() method
def train(self, text, vocab_size, verbose=False):
    # NEW: Split text into chunks first
    text_chunks = re.findall(self.pattern, text)
    encoded_chunks = [list(ch.encode('utf-8')) for ch in text_chunks]
    
    # CHANGE: Iterate over chunks, not single byte list
    for i in range(num_merges):
        stats = {}
        for chunk_ids in encoded_chunks:
            get_stats(chunk_ids, stats)  # Count pairs across all chunks
        pair = max(stats, key=stats.get)
        # ... merge logic
        encoded_chunks = [merge(chunk, pair, idx) for chunk in encoded_chunks]

# MODIFY: encode() method  
def encode(self, text):
    # NEW: Split into chunks, encode each separately
    text_chunks = re.findall(self.pattern, text)
    ids = []
    for chunk in text_chunks:
        chunk_bytes = chunk.encode('utf-8')
        chunk_ids = self._encode_chunk(chunk_bytes)  # Encode chunk independently
        ids.extend(chunk_ids)
    return ids

# ADD: Save/load methods
def save(self, path):
    with open(path, 'wb') as f:
        pickle.dump({
            'merges': self.merges,
            'vocabulary': self.vocabulary,
            'pattern': self.pattern.pattern  # Store pattern string
        }, f)

@classmethod
def load(cls, path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    tokenizer = cls()
    tokenizer.merges = data['merges']
    tokenizer.vocabulary = data['vocabulary']
    tokenizer.pattern = re.compile(data['pattern'])
    return tokenizer
```
























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

Unlike BasicTokenizer:
1. Add regex pattern splitting (GPT-4 style)
2. Split text into chunks before training
3. Train on chunks separately (no cross-chunk merging)
4. Add `save()` and `load()` methods
5. improved merge function

Key idea: Regex splits text into categories (words, numbers, punctuation) before BPE, preventing cross-category merges.
Why GPT-4 uses regex splitting
    Tokens are more useful for the model.
    Better generalization.
    Easier to interpret.




"""

def get_pairs(ids):
    counts = {}
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

class BasicTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.merges = {}
    def train(self, text, verbose=False):
        # encode the text
        # iterate over text, self.vocab_size - 256 times
        # count all of the pairs in a dictionary
        # choose the pair with the highest frequency
        # merge that pair as a new token
        # add that token to the vocab
        # {256: byte_string}
        # add to self.merges = {byte_string: 256}
        assert self.vocab_size > 256
        number_merges = self.vocab_size - 256
        byte_strings = text.encode('utf-8')
        ids = list(byte_strings)
        length_initial = len(ids)
        for i in range(number_merges):
            pairs = get_pairs(ids)
            pair = max(pairs, key=pairs.get)
            index = 256 + i
            ids = merge(ids, pair, index)
            self.merges[pair] = index
            self.vocabulary[index] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
        if verbose:
            length_final = len(ids)
            compression = length_initial/length_final
            print(length_initial, length_final)
            print(compression)
        # print(ids)
        
        
        # print(sorted([(v,k) for k,v in pairs.items()], reverse=True)[:10])
        # print(sorted(pairs.items(),reverse=True,key=lambda k: pairs[k])[:10])
        
    def encode(self, text):
        '''
        self.merges is important here
        
        we get text, and then we convert that text to byte strings, then to integers
        and then we iterate over the text until all pairs of merges that are
        *** we merge the pairs in the order they were merged at training ***
        possible under the trained tokenizer have been completed
        '''
        ids = list(text.encode('utf-8'))
        while len(ids) > 1:
            pairs = get_pairs(ids)
            '''
            pairs is a dictionary of tuples which tells us the frequency of each pair in the text to be encoded
            we dont' care about the frequency here, because we are not training
            we want to find the pair with the minimnunm index, THAT WAS MERGED
            key will take the key of pairs (which is the pair) we compare that pair against self.merges
            (32,32)
            '''
            pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids =  merge(ids, pair, self.merges[pair])
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

tokenizer = BasicTokenizer(300)
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokenizer.train(text, False)
print(tokenizer.encode('are hello'))
print(list('are hello'.encode('utf-8')))
print(tokenizer.decode(tokenizer.encode('are hello')))
# print(tokenizer.merges)

# print(tokenizer.decode([239, 188, 181, 239, 189, 142, 239, 189, 137, 239, 189, 131, 239, 189, 143, 239, 189, 132, 239, 189, 133, 33, 32, 263, 164, 263, 157, 263, 152, 263, 146, 263, 158, 263, 147, 263, 148, 258, 189]))