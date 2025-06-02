import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import regex as re
import pickle
import time
import matplotlib.pyplot as plt

# ========================
# Transformer Modules
# ========================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, emb, seq_len):
        super().__init__()
        pe = torch.zeros(seq_len, emb)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb, 2, dtype=torch.float) * (-torch.log(torch.tensor(1000.0)) / emb))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class LayerNormalization(nn.Module):
    def __init__(self, emb, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb))
        self.beta = nn.Parameter(torch.zeros(emb))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return self.gamma * (x - mean) / std + self.beta

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb, heads, mask):
        super().__init__()
        self.heads = heads
        self.head_dim = emb // heads
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)
        self.unify = nn.Linear(emb, emb)
        if mask:
            self.register_buffer("mask", torch.triu(torch.ones(128, 128), diagonal=1).bool())
        else:
            self.mask = None
    def forward(self, x):
        b, t, emb = x.size()
        q = self.toqueries(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.tokeys(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.tovalues(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.mask is not None:
            attn_scores = attn_scores.masked_fill(self.mask[:t, :t], float('-inf'))
        attn = torch.softmax(attn_scores, dim=-1)
        context = attn @ v
        context = context.transpose(1, 2).contiguous().view(b, t, emb)
        return self.unify(context)

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, ff_dim):
        super().__init__()
        self.attention = MultiHeadSelfAttention(emb, heads, mask)
        self.ff = nn.Sequential(nn.Linear(emb, ff_dim), nn.ReLU(), nn.Linear(ff_dim, emb))
        self.norm1 = LayerNormalization(emb)
        self.norm2 = LayerNormalization(emb)
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ff(self.norm2(x))

class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size, emb, heads, num_blocks, seq_len, ff_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb)
        self.positional_encoding = SinusoidalPositionalEncoding(emb, seq_len)
        self.blocks = nn.ModuleList([TransformerBlock(emb, heads, mask=True, ff_dim=ff_dim) for _ in range(num_blocks)])
        self.norm = LayerNormalization(emb)
        self.head = nn.Linear(emb, vocab_size)
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x[:, -1, :])
        return logits

# ========================
# Tokenizer (RegexTokenizer)
# ========================

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
    def __init__(self, text=None, vocab_size=None, pattern=None):
        if text is not None and vocab_size is not None:
            assert vocab_size > 256
            self.merges = {}
            self.vocabulary = {i: bytes([i]) for i in range(256)}
            self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
            self.compiled_pattern = re.compile(self.pattern)
            self.special_tokens = {"<pad>":256, "<unk>":257, "<sos>":258, "<eos>":259}
            for token, idx in self.special_tokens.items():
                self.vocabulary[idx] = token.encode('utf-8')
            self.train(text, vocab_size)
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
    def train(self, text, vocab_size, verbose=False):
        chunks = re.findall(self.compiled_pattern, text)
        encoded_chunks = [chunk.encode('utf-8') for chunk in chunks]
        num_merges = vocab_size - 256 - len(self.special_tokens)
        for i in range(num_merges):
            stats = {}
            for chunk in encoded_chunks:
                self.get_stats(chunk, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + len(self.special_tokens) + i
            encoded_chunks = [self.merge(chunk, pair, idx) for chunk in encoded_chunks]
            self.merges[pair] = idx
            self.vocabulary[idx] = self.vocabulary[pair[0]] + self.vocabulary[pair[1]]
    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            stats = {}
            self.get_stats(ids, stats)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
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
        return encoding
    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx in self.vocabulary:
                tokens.append(self.vocabulary[idx])
            else:
                tokens.append(self.vocabulary[self.special_tokens["<unk>"]])
        binary = b"".join(tokens)
        return binary.decode('utf-8', errors='replace')
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'merges': self.merges, 'vocabulary': self.vocabulary, 'pattern': self.pattern,
                         'compiled_pattern': self.compiled_pattern, 'special_tokens': self.special_tokens}, f)
    @classmethod
    def load(cls, path):
        tokenizer = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            tokenizer.merges = data['merges']
            tokenizer.vocabulary = data['vocabulary']
            tokenizer.pattern = data['pattern']
            tokenizer.compiled_pattern = data['compiled_pattern']
            tokenizer.special_tokens = data['special_tokens']
        return tokenizer

# ========================
# Dataset Class
# ========================

class HarryPotterDataset(Dataset):
    def __init__(self, file_paths, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.text = ""
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                self.text += f.read()
        self.tokens = self.tokenizer.encode(self.text)
    def __len__(self):
        return len(self.tokens) - self.seq_len
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ========================
# Text Generation Function
# ========================

def generate_text(model, tokenizer, seed_text, max_len=100, temperature=0.7, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(seed_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(tokens)
            next_token_logits = logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokenizer.decode(tokens.squeeze().tolist())

# ========================
# Main Training & Evaluation
# ========================

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_paths = ["book1.txt", "book2.txt", "book3.txt", "book4.txt", "book5.txt", "book6.txt", "book7.txt"]
    # For tokenizer training, we use text from one file (or concatenate a few) as sample
    with open(file_paths[0], "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokenizer = RegexTokenizer(text=sample_text, vocab_size=10000)
    dataset = HarryPotterDataset(file_paths, seq_len=128, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = AutoregressiveTransformer(vocab_size=10000, emb=512, heads=8, num_blocks=6, seq_len=128, ff_dim=2048).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    num_epochs = 10
    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y[:, -1])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
                losses.append(loss.item())
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss}")
        seed_text = "Harry Potter was"
        gen_text = generate_text(model, tokenizer, seed_text, max_len=100, device=device)
        print(f"Generated text after epoch {epoch}:\n{gen_text}")
    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()
    start_time = time.time()
    for batch_idx, (x, y) in enumerate(dataloader):
        pass
    batch_time = time.time() - start_time
    total_batches = len(dataloader) * num_epochs
    estimated_training_time = batch_time * total_batches
    print(f"Estimated training time: {estimated_training_time / 3600:.2f} hours")
