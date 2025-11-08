# # merge

# class BasicTokenizer:
#     def __init__(self, text, vocab_size):
#         assert vocab_size > 255
#         self.vocab_size = vocab_size
#     def train(self, text, vocab_size, verbose=False):
#     def encode(self, text):

#     def decode(self, ids):
#         pass


# principe Je vous le fais avec vous pour que vous puissiez apprendre à travers ma propre résolution de problèmes.
# Je vais également écrire le pseudo-code pour chaque fonction avant de le faire.
# La philosophie est de montrer toutes les choses dans des petits exemples aussi


# [1, 255, 242, 0, 34]

# (1,255): 1, (255, 242): 1…

def get_pairs(ids):
    counts = {}
    for a,b in zip(ids,ids[1:]):
        counts[(a,b)] = counts.get((a,b), 0) + 1
    return counts

def merge(ids, merge_index, pair):
    """
    this function takes a list of ids which are the encoded byte strings of a text
    it then finds all instances pair, and replaces them with merge_index, then returns the new list
    I is not tracking the place in our new list. It's tracking the place in the IDs list which is to be updated
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(merge_index)
            i+=2
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

class BasicTokenizer:
    def __init__(self, vocab_size):
        assert vocab_size > 256
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab= {index : bytes([index]) for index in range(256)}


    def train(self, text):

        # looping over our vocab size to do vocab_size - 256 merges
        # encode our text into byte strings
        # for each merge

        # we count the number of pairs

        # find the highest frequency pair

        # merge it in the list of ids

        # update self.merges and self.ids
        merges = self.vocab_size - 256
        ids = list(text.encode('utf-8'))
        for i in range(merges):
            pairs = get_pairs(ids)
            merge_index = i + 255
            pair = max(pairs, key=pairs.get)
            ids = merge(ids,merge_index, pair)
            self.merges[pair] = merge_index
            self.vocab[merge_index] = self.vocab[pair[0]] + self.vocab[pair[1]]
    
    def encode(self, text):
        # encode input is raw text
        # first step is to encode utf-8
        ids = list(text.encode('utf-8'))
        while len(ids) > 1:
            # this gives us all the pairs in the text, we don't care abotu frequency for now
            pairs = get_pairs(ids)
            # pick the merge_pair with the lowest id since we need to do it in the order they were merged
            # so we pick from pairs, and any pair that is not in merges will be "inf" and we take min so it's gone
            # otherwise, we have self.merges.get(pair), if it's in the merges dict, we get the merge_index
            # the pair with the lowest merge index is chosen
            pair = min(pairs,key = lambda p: self.merges.get(p, "inf"))
            # the pair check here serves to say: if all pairs checked were not in self.merges
            # and therefore it was all "inf", then a random pair will be picked, meaning
            # that in the pairs dict, there are no more pairs that map to a merge we have done in training
            # therefore, we should quit since theere are no more pairs left to merge (no pair found that is also in self.merges)
            if pair not in self.merges:
                break
            ids = merge(ids, self.merges[pair], pair)
            return ids


        

    def decode(self,ids):
        ids = [self.vocab[i] for i in ids]
        byte_strings = b''.join(ids)
        ids = byte_strings.decode('utf-8')


tokenizer = BasicTokenizer(258)
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
tokenizer.train(text)


print(tokenizer.merges, sep="\n\n\n")
#     def encode(text):
#         pass

#     def decode(bytes):
#         pass














# def get_stats(ids, counts=None):
#     counts = counts if counts else {}
#     for pair in (zip(ids, ids[1:])):
#         counts[pair] = counts.get(pair, 0) + 1
#     return counts


# def merge(ids, pair, idx):
#     """
#     In the list of integers (ids), replace all consecutive occurrences
#     of pair with the new integer token idx
#     Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
#     """
#     newids = []
#     i = 0
#     while i < len(ids):
#         # if not at the very last position AND the pair matches, replace it
#         if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
#             newids.append(idx)
#             i += 2
#         else:
#             newids.append(ids[i])
#             i += 1
#     return newids


# class BasicTokenizer:

#     def __init__(self):
#         # default: vocab size of 256 (all bytes), no merges, no patterns
#         self.merges = {} # (int, int) -> int
#         self.vocab = {idx: bytes([idx]) for idx in range(256)}

#     def train(self, text, vocab_size, verbose=False):
#         assert vocab_size > 255
#         merges = vocab_size - 255
#         char_bytes = text.encode('utf-8')
#         ids = list(map(int, char_bytes))
#         for i in range(merges):
#             counts = get_stats(ids)
#             # print(sorted([(v,k) for k,v in counts.items()], reverse=True))
#             sorted(counts.items(), key=lambda key_value:key_value[1],reverse=True)
#             pair = max(counts, key=counts.get)
#             # print(pair)
#             index = 256 + i
#             ids = merge(ids,pair,index)
#             # print(ids)
#             self.merges[pair] = index
#             self.vocab[index] = self.vocab[pair[0]] + self.vocab[pair[1]]
#             if verbose:
#                 print(f'merged pair {pair} with index {index}, {pair} had {counts[pair]} occurences')

#     def decode(self, ids):
#         text_bytes = b''.join(self.vocab[i] for i in ids)
#         return text_bytes.decode("utf-8")


#     def encode(self, text):
#         # steps
#         # convert to bytes
#         text_bytes = text.encode("utf-8")
#         # If we were not doing BPE, we would be finished here.
#         ids = list(text_bytes)
#         while len(ids) > 1:
#             # we only need the pairs, the frequencies are unnecesary
#             counts = get_stats(ids)
#             merge_pair = min(counts, key = lambda pair: self.merges.get(pair, float("inf")))
#             if merge_pair not in self.merges:
#                 break
#             idx = self.merges[merge_pair]
#             ids = merge(ids, merge_pair, idx)
#         return ids

# tokenizer = BasicTokenizer()
# text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
# tokenizer.train(text,260)

# # print(tokenizer.decode(tokenizer.encode("hello world")))

# print((list(tokenizer.vocab.items())[256:]), tokenizer.merges, sep="\n\n\n")