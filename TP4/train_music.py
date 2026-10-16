import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Reproductibilité
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/mnt/ssd/irishman"

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data   = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Nombre de chansons (Train) : {len(train_data)}")
print(f"Nombre de chansons (Val)   : {len(val_data)}")
print("\n--- Exemple de partition (Train[0]) ---")
print(train_data[0]["abc notation"])

# ===== Étape 2 : vocabulaire et padding =====

# 1. Extraction déterministe (triée) des caractères uniques
all_text = "".join(song["abc notation"] for song in train_data + val_data)
unique_chars = sorted(list(set(all_text)))
vocab_size = len(unique_chars)
print(f"Taille du vocabulaire : {vocab_size} caractères uniques")
print(f"Caractères : {repr(''.join(unique_chars))}")

# 2. Mappings caractère <-> index
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

def vectorize_string(s):
    return [char_to_index[char] for char in s]

# 3. Statistiques de longueur et Padding
lengths = [len(song["abc notation"]) for song in train_data]
max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)
print(f"Longueur moyenne : {avg_len:.1f} | Longueur max : {max_len}")

SEQ_LENGTH = 500  # Longueur fixe choisie (environ 2x la longueur moyenne)

assert " " in char_to_index, "l'espace doit être dans le vocabulaire"

def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) >= max_length:
        return sequence[:max_length]
    return sequence + " " * (max_length - len(sequence))

# Vérification
s = train_data[0]["abc notation"]
padded = pad_or_truncate(s)
print(f"Partition 0 : {len(s)} caractères -> {len(padded)} après padding/troncature")
print(f"Vectorisée (10 premiers indices) : {vectorize_string(padded)[:10]}")

n_trunc = sum(1 for l in lengths if l > SEQ_LENGTH)
print(f"Partitions tronquées : {n_trunc} / {len(lengths)} ({100*n_trunc/len(lengths):.1f} %)")

# ===== Exercice 2 : Dataset et modèle =====

class MusicDataset(Dataset):
    def __init__(self, raw_data, seq_length=SEQ_LENGTH, subset_size=None):
        if subset_size is not None:
            raw_data = raw_data[:subset_size]

        self.data = [
            vectorize_string(pad_or_truncate(song["abc notation"], seq_length))
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # Entrée : tous les caractères sauf le dernier
        x = torch.tensor(seq[:-1], dtype=torch.long)
        # Cible : tous les caractères sauf le premier (décalage de +1)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class MusicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type doit être 'rnn' ou 'lstm'")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x : (batch_size, seq_len)
        embeds = self.embedding(x)                          # (batch_size, seq_len, embedding_dim)
        out, hidden_state = self.rnn(embeds, hidden_state)  # (batch_size, seq_len, hidden_size)
        logits = self.fc(out)                               # (batch_size, seq_len, vocab_size)
        return logits, hidden_state


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# ===== Exercice 3 : entraînement =====

@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn, vocab_size):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(x)

        # Aplatissement (batch_size * seq_len, vocab_size) pour CrossEntropyLoss
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))

        total_loss += loss.item() * y.numel()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens  += y.numel()

    return total_loss / total_tokens, total_correct / total_tokens


def train_model(
    rnn_type="lstm",
    subset_size=10000,
    num_epochs=15,
    batch_size=128,
    lr=5e-3,
    embedding_dim=256,
    hidden_size=512,
    max_grad_norm=None,
    patience=5,
):
    train_dataset = MusicDataset(train_data, subset_size=subset_size)
    val_dataset   = MusicDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = MusicModel(vocab_size, embedding_dim, hidden_size, rnn_type=rnn_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    clip_tag = f"clip{max_grad_norm}" if max_grad_norm is not None else "noclip"
    run_name = f"{rnn_type}_sub{subset_size}_lr{lr}_{clip_tag}_{time.strftime('%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs_rnn/{run_name}")
    save_path = f"best_{rnn_type}.pth"

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss, total_train_tokens = 0.0, 0
        epoch_grad_norms = []

        for x, y in tqdm(train_loader, desc=f"[{run_name}] Epoch {epoch}/{num_epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()

            # Norme des gradients (et clipping éventuel AVANT optimizer.step())
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

            epoch_grad_norms.append(grad_norm.item())
            if global_step % 20 == 0:
                writer.add_scalar("GradNorm/step", grad_norm.item(), global_step)

            optimizer.step()

            total_train_loss += loss.item() * y.numel()
            total_train_tokens += y.numel()
            global_step += 1

        avg_train_loss = total_train_loss / total_train_tokens
        avg_grad_norm  = sum(epoch_grad_norms) / len(epoch_grad_norms)
        val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, vocab_size)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("GradNorm/Epoch_Avg", avg_grad_norm, epoch)
        writer.flush()

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Avg GradNorm: {avg_grad_norm:.3f}", flush=True)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping déclenché à l'époque {epoch}.")
                break

    writer.close()
    return model


# ===== Exercice 4 : génération =====

@torch.no_grad()
def generate_music(model, start_sequence, output_length=300, temperature=1.0):
    model.eval()
    generated_seq = start_sequence

    # 1. Pré-remplissage (lecture de l'amorce)
    input_indices = [char_to_index[c] for c in start_sequence]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)
    logits, hidden_state = model(input_tensor, hidden_state=None)

    # 2. Boucle auto-régressive
    for _ in range(output_length):
        last_logits = logits[:, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1).squeeze(0)

        next_index = torch.multinomial(probs, num_samples=1).item()
        next_char = index_to_char[next_index]
        generated_seq += next_char

        input_tensor = torch.tensor([[next_index]], dtype=torch.long, device=device)
        logits, hidden_state = model(input_tensor, hidden_state=hidden_state)

    return generated_seq


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    cfg = dict(subset_size=2000, num_epochs=1) if quick else dict(subset_size=50000, num_epochs=30)

    train_model(rnn_type="lstm", hidden_size=1024, max_grad_norm=1.0, patience=5, **cfg)

    # Rechargement du meilleur modèle (Early Stopping)
    model = MusicModel(vocab_size, 256, 1024, rnn_type="lstm").to(device)
    model.load_state_dict(torch.load("best_lstm.pth", map_location=device))

    prompt = "X:1\nT:Irish Reel\nM:4/4\nL:1/8\nK:D\n"
    with open("generated_abc.txt", "w", encoding="utf-8") as f:
        for T in [0.2, 0.8, 2.0]:
            abc = generate_music(model, prompt, output_length=300, temperature=T)
            print(f"\n===== T = {T} =====\n{abc}\n", flush=True)
            f.write(f"===== T = {T} =====\n{abc}\n\n")
