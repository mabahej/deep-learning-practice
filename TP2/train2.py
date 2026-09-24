import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, accuracy_score
from dataset import CardioDataset

# Datasets et DataLoaders (même découpage que dans dataset.py)
dataset = CardioDataset("data/cardio_train.csv")
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

batch = next(iter(train_loader))

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() # Sortie binaire [0, 1]
        )

    def forward(self, x):
        return self.net(x)

# Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=batch['features'].shape[1], hidden_size=128).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 0.1
l2_lambda = 0

for epoch in range(10):
    model.train()

    all_targets = []
    all_predictions = []

    for batch in train_loader:
        inputs, targets = batch["features"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        base_loss = criterion(outputs, targets)

        # Calcul de la pénalité L1 (somme des valeurs absolues des poids)
        l1_penalty = sum(p.abs().sum() for p in model.parameters())

        # Calcul de la pénalité L2 (somme des carrés des poids)
        l2_penalty = sum(p.pow(2).sum() for p in model.parameters())

        # Loss totale
        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        loss.backward()
        optimizer.step()

        predictions = (outputs >= 0.5).float()

        all_targets.extend(targets.detach().cpu().numpy().ravel())
        all_predictions.extend(predictions.detach().cpu().numpy().ravel())

    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)

    print(
        f"Epoch {epoch + 1}/10 - "
        f"Loss: {loss.item():.4f} - "
        f"Accuracy: {accuracy:.4f} - "
        f"Precision: {precision:.4f}"
    )


# Évaluation sur le test set
model.eval()

all_targets = []
all_predictions = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch["features"].to(device)
        targets = batch["labels"].to(device)

        outputs = model(inputs)
        predictions = (outputs >= 0.5).float()

        all_targets.extend(targets.cpu().numpy().ravel())
        all_predictions.extend(predictions.cpu().numpy().ravel())

test_accuracy = accuracy_score(all_targets, all_predictions)
test_precision = precision_score(
    all_targets,
    all_predictions,
    zero_division=0
)

print("\n--- Test Results ---")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
