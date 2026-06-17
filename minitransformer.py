import torch
import torch.nn as nn
from torch.nn import functional as F

# =====================================================
# LOAD DATA
# =====================================================

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("Vocabulary size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join([itos[i] for i in ids])

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

print("Dataset length:", len(data))

# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# =====================================================
# HYPERPARAMETERS
# =====================================================

batch_size = 32
block_size = 64

max_iters = 2000
learning_rate = 3e-4

eval_interval = 200

n_embd = 128
n_head = 4
n_layer = 4

dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =====================================================
# BATCH FUNCTION
# =====================================================

def get_batch(split):

    data_source = train_data if split == "train" else val_data

    max_start = len(data_source) - block_size - 1

    if max_start <= 0:
        raise ValueError(
            f"{split} dataset too small.\n"
            f"length={len(data_source)}\n"
            f"block_size={block_size}"
        )

    ix = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack(
        [data_source[i:i+block_size] for i in ix]
    )

    y = torch.stack(
        [data_source[i+1:i+block_size+1] for i in ix]
    )

    return x.to(device), y.to(device)

# =====================================================
# LOSS ESTIMATION
# =====================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    out = {}

    for split in ["train", "val"]:

        losses = torch.zeros(20)

        for k in range(20):

            X, Y = get_batch(split)

            _, loss = model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

# =====================================================
# ATTENTION HEAD
# =====================================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)

        wei = wei * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim=-1
        )

        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out

# =====================================================
# MULTI HEAD ATTENTION
# =====================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(
            num_heads * head_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )

        out = self.proj(out)

        return self.dropout(out)

# =====================================================
# FEED FORWARD
# =====================================================

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# =====================================================
# TRANSFORMER BLOCK
# =====================================================

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedForward(
            n_embd
        )

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )

    def forward(self, x):

        x = x + self.sa(
            self.ln1(x)
        )

        x = x + self.ffwd(
            self.ln2(x)
        )

        return x

# =====================================================
# GPT MODEL
# =====================================================

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding_table(
            idx
        )

        pos_emb = self.position_embedding_table(
            torch.arange(
                T,
                device=device
            )
        )

        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, _ = self(
                idx_cond
            )

            logits = logits[:, -1, :]
            temperature = 0.8

            probs = F.softmax(logits / temperature,dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx

# =====================================================
# CREATE MODEL
# =====================================================

model = GPTLanguageModel().to(device)

print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

# =====================================================
# TRAIN ONLY WHEN RUN DIRECTLY
# =====================================================

if __name__ == "__main__":

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    # =====================================================
    # TRAINING LOOP
    # =====================================================

    for step in range(max_iters):

        if step % eval_interval == 0:

            losses = estimate_loss()

            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        # SAVE CHECKPOINT EVERY 500 STEPS
        if step % 500 == 0 and step > 0:

            torch.save(
                model.state_dict(),
                f"checkpoint_{step}.pth"
            )

            print(f"Saved checkpoint_{step}.pth")

        xb, yb = get_batch("train")

        logits, loss = model(
            xb,
            yb
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()
# =====================================================
# SAVE MODEL
# =====================================================

torch.save(
    model.state_dict(),
    "mini_gpt.pth"
)

print("Model saved.")

# =====================================================
# GENERATE TEXT
# =====================================================

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=1000
)

print("\nGenerated text:\n")
print(decode(generated[0].tolist()))