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
        mergeable_ranks = {}
        # Add base 256 bytes
        for i in range(256):
            mergeable_ranks[bytes([i])] = i
        # Add merged tokens (reconstruct bytes in order)
        token_bytes = {i: bytes([i]) for i in range(256)}
        sorted_merges = sorted(tokenizer.merges.items(), key=lambda x: x[1])
        for (pair, merged_id) in sorted_merges:
            merged_bytes = token_bytes[pair[0]] + token_bytes[pair[1]]
            token_bytes[merged_id] = merged_bytes
            mergeable_ranks[merged_bytes] = merged_id
        return mergeable_ranks

class NanoChatTokenizer:

    """Light wrapper around tiktoken (for efficient inference) but train with python tokenizer"""

    # We store the BOS id because it’s the one special token the runtime always needs and must be mapped correctly across encodings.
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text, vocab_size, pattern):
        # Step 1: Train Python tokenizer (NO special token handling needed!)
        tokenizer = RegexTokenizer(pattern)
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train(text, vocab_size_no_special)
        # Step 2: Convert to mergeable_ranks
        mergeable_ranks = get_mergeable_ranks(tokenizer)
        # Step 3: Add special tokens (tiktoken handles everything!)
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i 
            for i, name in enumerate(SPECIAL_TOKENS)
        }
        # Step 4: Create tiktoken.Encoding
        enc = tiktoken.Encoding(
            name="nanochat",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,  # ← That's it! Tiktoken handles the rest
        )
        return cls(enc, "<|bos|>")
        
    # encode(allowed_special="all") only allows special tokens that are already in the text. Nanochat's approach is safer and more explicit.
    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids
    
    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)
    
    def decode(self, ids):
        # Tiktoken handles both regular and special tokens automatically!
        return self.enc.decode(ids)

    # show demo
    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

text = r"""
(Washington, D.C., July 9, 2025)- Yesterday, Mexico’s National Service of Agro-Alimentary Health, Safety, and Quality (SENASICA) reported a new case of New World Screwworm (NWS) in Ixhuatlan de Madero, Veracruz in Mexico, which is approximately 160 miles northward of the current sterile fly dispersal grid, on the eastern side of the country and 370 miles south of the U.S./Mexico border. This new northward detection comes approximately two months after northern detections were reported in Oaxaca and Veracruz, less than 700 miles away from the U.S. border, which triggered the closure of our ports to Mexican cattle, bison, and horses on May 11, 2025.

While USDA announced a risk-based phased port re-opening strategy for cattle, bison, and equine from Mexico beginning as early as July 7, 2025, this newly reported NWS case raises significant concern about the previously reported information shared by Mexican officials and severely compromises the outlined port reopening schedule of five ports from July 7-September 15. Therefore, in order to protect American livestock and our nation’s food supply, Secretary Rollins has ordered the closure of livestock trade through southern ports of entry effective immediately.

“The United States has promised to be vigilant — and after detecting this new NWS case, we are pausing the planned port reopening’s to further quarantine and target this deadly pest in Mexico. We must see additional progress combatting NWS in Veracruz and other nearby Mexican states in order to reopen livestock ports along the Southern border,” said U.S. Secretary of Agriculture Brooke L. Rollins. “Thanks to the aggressive monitoring by USDA staff in the U.S. and in Mexico, we have been able to take quick and decisive action to respond to the spread of this deadly pest.”
""".strip()

tokenizer = NanoChatTokenizer.train_from_iterator(text, 300, r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

print(tokenizer.decode(tokenizer("hello")))