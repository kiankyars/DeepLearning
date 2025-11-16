def save(self, tokenizer_dir):
    """Save tiktoken.Encoding to disk as pickle."""
    os.makedirs(tokenizer_dir, exist_ok=True)
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(self.enc, f)
    print(f"Saved tokenizer encoding to {pickle_path}")

@classmethod
def from_directory(cls, tokenizer_dir):
    """Load tokenizer from disk."""
    pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
    with open(pickle_path, "rb") as f:
        enc = pickle.load(f)
    return cls(enc, "<|bos|>")

# **Demo**: Save tokenizer, load it back, verify encode/decode works.

#### **1.3 Info getters**
def get_vocab_size(self):
    return self.enc.n_vocab

def get_special_tokens(self):
    return self.enc.special_tokens_set

# **Demo**: Print vocab size, list special tokens
@classmethod
def from_pretrained(cls, tiktoken_name):
    """Load pretrained tiktoken (e.g., 'cl100k_base' for GPT-4)."""
    enc = tiktoken.get_encoding(tiktoken_name)
    # Pretrained uses "<|endoftext|>" not "<|bos|>"
    return cls(enc, "<|endoftext|>")

# **Demo**: Compare vocab sizes (GPT-2 vs GPT-4 vs yours)

# **Part 2: Conversation rendering (10-12 min)**

# **2.1 Conceptual explanation (3-4 min)**

# Explain the problem:
# - Chat format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
# - Need: Token IDs + training mask (1 = predict, 0 = ignore)
# - Special tokens: `<|user_start|>`, `<|user_end|>`, `<|assistant_start|>`, `<|assistant_end|>`

# Show example:
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
# (only assistant content is masked = 1)
# - Why this works:
# - The model sees user text as context, so it can condition on it.
# - The model only learns to predict assistant tokens, which is the goal.

# Model generates: <|python_start|> 'hello'.count('l') <|python_end|>
# System executes the code: use_calculator(expr)
# System forces the result: <|output_start|> 2 <|output_end|>
# Model continues generating after the output

# Training data:
{"type": "python", "text": "123 + 456"}      # mask=1 (model predicts this)
{"type": "python_output", "text": "579"}     # mask=0 (system provides this)

# At inference:
# Model predicts: "<|python_start|> 123 + 456 <|python_end|>"
# System runs it: result = 579
# System injects: "<|output_start|> 579 <|output_end|>"
# Model continues: (sees the output, continues reasoning)

#### **2.4 render_for_completion() (2 min)**

def render_for_completion(self, conversation):
    """For RL: remove last assistant message, append <|assistant_start|>."""
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    assert messages[-1]["role"] == "assistant"
    messages.pop()  # Remove last message
    
    ids, mask = self.render_conversation(conversation)
    assistant_start = self.encode_special("<|assistant_start|>")
    ids.append(assistant_start)
    return ids

# **Demo**: Show before/after - conversation with answer, then primed for generation.













### **Part 3: Convenience functions (3-5 min)**

#### **3.1 get_tokenizer()**

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)

# Purpose: Loads the tokenizer from the standard location (`~/.cache/nanochat/tokenizer/`). Used throughout the codebase instead of hardcoding paths.

#### **3.2 get_token_bytes()**

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found..."
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes

# Purpose: Loads a precomputed tensor `token_bytes[vocab_size]` where `token_bytes[token_id]` = number of bytes that token represents (0 for special tokens).

# This normalizes by actual byte length, so you can compare tokenizers with different vocab sizes.

# key-insight: bytes are the invariant—both tokenizers compress the same original bytes, so normalize by that.