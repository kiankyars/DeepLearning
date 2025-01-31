import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Assuming you have a dataset class and dataloader
class HarryPotterDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        self.tokens = self.tokenize(text)

    def tokenize(self, text):
        # Implement tokenization (e.g., word-level or subword-level)
        return tokenizer(text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Load dataset
text = load_harry_potter_text()  # Implement this function
dataset = HarryPotterDataset(text, seq_len=128)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = AutoregressiveTransformer(vocab_size=10000, emb=512, heads=8, num_blocks=6, seq_len=128, ff_dim=2048)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)  # Shape: (batch_size, seq_len, vocab_size)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    print(f"Epoch {epoch}, Average Loss: {total_loss / len(dataloader)}")


def generate_text(model, seed_text, max_len=100, temperature=0.7):
    model.eval()
    tokens = tokenizer(seed_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(tokens)  # Shape: (1, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    generated_text = tokenizer.decode(tokens.squeeze().tolist())
    return generated_text

# Example usage
seed_text = "Harry Potter was"
generated_text = generate_text(model, seed_text, max_len=100)
print(generated_text)