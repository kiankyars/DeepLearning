import pickle
import regex as re

class RegexTokenizer:
    def __init__(self, vocab_size=None, text=None, pattern=None) -> None:
        if not(vocab_size and text and pattern):
            return
        assert vocab_size > 256
        self.pattern = re.compile(pattern)
        self.merges = {}
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.train(text, vocab_size)
        
    def train(self, text, vocab_size) -> None:
        split_text = re.findall(self.pattern, text)
        encoded_split_text = [chunk.encode('utf-8') for chunk in split_text]
        merges = vocab_size - 256
        for i in range(merges):
            stats = {}
            for chunk in encoded_split_text:
                self._get_stats(stats, chunk)
            id = i + 256
            pair = max(stats, key=stats.get)
            # print(pair)
            self.merges[pair] = id
            self.vocabulary[id] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
            encoded_split_text = [self._merge(chunk, pair, id) for chunk in encoded_split_text]
    def encode(self, text) -> None:
        # receives text => binary => numbers
        split_text = re.findall(self.pattern, text)
        encoding = []
        for chunk in split_text:
            byte_text = chunk.encode('utf-8')
            code_points = list(byte_text)
            merged_encoding = self._encode_chunk(code_points)
            encoding.extend(merged_encoding)
        return encoding
    def _encode_chunk(self, code_points) -> None:
        # receives a chunk of code_points and must merge them w/ the BPE
        while len(code_points) > 1:
            stats = {}
            self._get_stats(stats, code_points)
            pair = min(stats, key=lambda x: self.merges.get(x, float('inf')))
            if pair not in self.merges:
                break
            code_points = self._merge(code_points, pair, self.merges[pair])
        # merge is complete
        return code_points
    def decode(self, code_points) -> None:
        # receives a list of utf-8 code_points w/ merges done to them and must convert them to binary and then text
        # this is the single line I refer to in my remnote
        # first step is to convert to bytes with list compreension, then do binary
        # string join, followed by decode
        return (b"".join([self.vocabulary[code_point] for code_point in code_points])).decode('utf-8')
        # I thought the above should be:
        # return (b"".join([bytes([i]) for i in code_points])).decode('utf-8')

    def _get_stats(self, stats, ids) -> None:
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
    def _merge(self, code_points, pair, id) -> None:
        i = 0
        merged_code_points = []
        while i < len(code_points):
            if i < len(code_points) - 1 and code_points[i] == pair[0] and code_points[i + 1] == pair[1]:
                merged_code_points.append(id)
                i += 2
            else:
                merged_code_points.append(code_points[i])
                i += 1
        return merged_code_points
    def save(self, path) -> None:
        with open(path, 'wb') as f:
            pickle.dump({
                'merges': self.merges,
                'pattern': self.pattern,
                'vocabulary': self.vocabulary
            }, f)

    @classmethod
    def load(cls, path) -> None:
        tokenizer = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            tokenizer.merges = data['merges']
            tokenizer.pattern = data['pattern']
            tokenizer.vocabulary = data['vocabulary']
        return tokenizer