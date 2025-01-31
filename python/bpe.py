GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
import regex as re
import pickle

class RegexTokenizer:
    def __init__(self, text=None, vocab_size=None, pattern=None):
        if text is not None and vocab_size is not None:
            self.text = text
            self.vocab_size = vocab_size
            self.merges = {}
            self.vocabulary = {i: bytes([i]) for i in range(256)}
            self.pattern = GPT4_SPLIT_PATTERN if not pattern else pattern
            self.compiled_pattern = re.compile(self.pattern)
            self.train()

    def get_stats(self, ids, stats=None):
        counts = stats if stats is not None else {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, verbose=False):
        '''
        perform BPE over a given text
        by looping over and merging most common pairs
        until the desired vocab_size is reaches
        input: text to tokenize
        output: merges dictionary and vocabulary dictionary
        '''
        chunks = re.findall(self.compiled_pattern, self.text)
        # for i in range(len(chunks)):
        #     chunks[i] = list(chunks[i].encode('utf-8'))
        encoded_chunks = [chunk.encode('utf-8') for chunk in chunks]
        num_merges = self.vocab_size - 256
        # ids = list(tokens) # copy so we don't destroy the original list
        # iterate over merges
        # print(tokens)
        stats = {}
        for i in range(num_merges):
            # get the consecutive token stats at time step i
            # print(tokens)
            # dictionary of all the co-occurence frequencies
            for chunk in encoded_chunks:
                self.get_stats(chunk, stats)
            # print(stats)
            # return
            # call max on stats with key as the dictionary get function to find the highest pair co-occurence
            pair = max(stats, key=stats.get)
            # iterate our new index, i = 0 at start, so we have idx = 256
            idx = 256 + i
            # give statistics
            if verbose:
                print(f"merging {pair} into a new token {idx}")
            # call merge function on the pair with new idx, and give ids list so it can be modified
            encoded_chunks = [self.merge(chunk, pair, idx) for chunk in encoded_chunks]
            # update merges dictionary
            self.merges[pair] = idx
            # this is valid, but less effecient
            # self.vocabulary[v] = bytes([k[0]]) + bytes([k[1]])
            # better
            self.vocabulary[idx] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]

    def _encode_chunk(self, ids):
        # receives ids and then goes through merge dictionary to add merges
        while len(ids) >= 2:
            stats = {}
            self.get_stats(ids, stats)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            # self.merges or self.merges.keys
            if pair not in self.merges:
                break
            ids = self.merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        byte_texts = re.findall(self.compiled_pattern, text)
        encoding = []
        for chunk in byte_texts:
            byte_text = chunk.encode('utf-8')
            ids = list(map(int, byte_text))
            merged_ids = self._encode_chunk(ids)
            encoding.extend(merged_ids)
        # chunked_texts = [chunk.encode('utf-8') for chunk in byte_texts]
        # encoding = [self._encode_chunk(ids) for ids in chunked_texts]
        return encoding

    def decode(self, ids):
        '''
        given a list of integers, return the strings, en passant par la binaire
        '''
        # self.vocabulary will map our ints to bytes so they can be merged w/
        # b"".join() and then decoded
        binary = b"".join([self.vocabulary[i] for i in ids])
        return binary.decode('utf-8')
    
    def save(self, path):
        """Save the trained tokenizer to a file"""
        # binary
        with open(path, 'wb') as f:
            pickle.dump({
                'merges': self.merges,
                'vocabulary': self.vocabulary,
                'pattern': self.pattern,
                'compiled_pattern': self.compiled_pattern
            }, f)

    @classmethod
    def load(cls, path):
        """Load a trained tokenizer from a file"""
        # binary
        tokenizer = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            tokenizer.merges = data['merges']
            tokenizer.vocabulary = data['vocabulary']
            tokenizer.pattern = data['pattern']
            tokenizer.compiled_pattern = data['compiled_pattern']
        return tokenizer

text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
# print(type(list(text.encode("utf-8"))[0])) # <class 'int'>
tokenizer = RegexTokenizer(text, 276)
tokenizer.save('/Users/kian/Code/DeepLearning/models/tokenizer.pkl')