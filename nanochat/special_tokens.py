class RegexTokenizer:
    def __init__(self, pattern=None):
        self.pattern = re.compile(pattern or GPT4_SPLIT_PATTERN)
        self.merges = {}
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        # NEW: Special token storage
        self.special_tokens = {}  # str -> int
        self.inverse_special_tokens = {}  # int -> str

    # NEW: Register special tokens (call after training or before using)
    def register_special_tokens(self, special_tokens):
        """
        special_tokens: dict[str, int] e.g. {"<|endoftext|>": 100257}
        """
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
        # Note: We don't add them to vocabulary - they're handled separately

    # MODIFY: train() - reserve space for special tokens
    def train(self, text, vocab_size, verbose=False):
        # NEW: Account for special tokens
        # If you plan to add 8 special tokens later, train fewer merges
        num_special = len(self.special_tokens) if self.special_tokens else 0
        num_merges = vocab_size - 256 - num_special
        # 256 + num_special + num_merges = vocab_size
        
        # ... rest of training (unchanged)
        # After training, special tokens get IDs starting at (256 + num_merges)

    # NEW: Encode ordinary text (ignores special tokens)
    def encode_ordinary(self, text):
        """Encode text, treating any special tokens as regular text."""
        text_chunks = re.findall(self.pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode('utf-8')
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    # MODIFY: encode() - handle special tokens
    def encode(self, text, allowed_special="none_raise"):
        """
        allowed_special: "all" | "none" | "none_raise" | set of token strings
        - "none_raise": Raise error if special token found (default, safest)
        - "none": Ignore special tokens (treat as regular text)
        - "all": Allow all special tokens
        - set: Allow only tokens in the set
        """
        # Determine which special tokens to allow
        if allowed_special == "none_raise":
            # Check if any special token is in text
            for token in self.special_tokens:
                if token in text:
                    raise ValueError(f"Special token {token} found in text")
            return self.encode_ordinary(text)
        elif allowed_special == "none":
            return self.encode_ordinary(text)
        elif allowed_special == "all":
            allowed = self.special_tokens
        elif isinstance(allowed_special, set):
            allowed = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        # NEW: Split text on special tokens
        # Create regex pattern to match any allowed special token
        special_pattern = "(" + "|".join(re.escape(k) for k in allowed.keys()) + ")"
        parts = re.split(special_pattern, text)
        
        ids = []
        for part in parts:
            if part in allowed:
                # This part is a special token
                ids.append(allowed[part])
            else:
                # This part is regular text
                ids.extend(self.encode_ordinary(part))
        return ids

    # MODIFY: decode() - handle special token IDs
    def decode(self, ids):
        """Decode token IDs, handling both regular and special tokens."""
        part_bytes = []
        for idx in ids:
            if idx in self.vocabulary:
                # Regular token: get bytes from vocabulary
                part_bytes.append(self.vocabulary[idx])
            elif idx in self.inverse_special_tokens:
                # Special token: get string, encode to bytes
                special_str = self.inverse_special_tokens[idx]
                part_bytes.append(special_str.encode('utf-8'))
            else:
                raise ValueError(f"Invalid token ID: {idx}")
        
        text_bytes = b"".join(part_bytes)
        return text_bytes.decode('utf-8', errors='ignore')

    # NEW: Helper to encode a single special token
    def encode_special(self, text):
        """Encode a single special token string to its ID."""
        if text not in self.special_tokens:
            raise ValueError(f"Unknown special token: {text}")
        return self.special_tokens[text]

tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size=1000)

# Register special tokens AFTER training
# IDs must be >= vocab_size (e.g., if vocab_size=1000, use 1000+)
tokenizer.register_special_tokens({
    "<|endoftext|>": 1000,
    "<|bos|>": 1001,
})

# Now encoding handles special tokens
ids = tokenizer.encode("Hello <|bos|> world", allowed_special="all")
# Returns: [regular tokens for "Hello "], [1001], [regular tokens for " world"]