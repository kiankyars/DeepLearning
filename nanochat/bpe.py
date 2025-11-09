"""
SERIES PRINCIPLES
==========================

I'm doing this with you so you can learn through my own problem-solving process.
I will write pseudocode for each function before implementing it.
I demonstrate everything through small examples.
I'm going to do this in one take.
rust, ratio change, constantly shifting

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
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