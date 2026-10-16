import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# Exercice 1 : prétraitement et augmentation de données

# 1. Pipeline d'entraînement (avec Data Augmentation)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 2. Pipeline de validation (déterministe)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(batch_size=256, n_workers=8, subset_size=None):
    print("Chargement d'ImageNet...")
    train_dataset = ImageNet(root=data_dir, split="train", transform=train_transform)
    val_dataset = ImageNet(root=data_dir, split="val", transform=val_transform)

    # Optionnel : sous-ensemble pour tester rapidement le code en séance
    if subset_size is not None:
        generator = torch.Generator().manual_seed(42)
        train_indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size]
        val_indices = torch.randperm(len(val_dataset), generator=generator)[:subset_size // 4]
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader



# Exercice 2 : architectures


class MLP(nn.Module):
    def __init__(self, input_size=3 * 227 * 227, hidden_size=256, num_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetDeep(AlexNet):
    """Question 3.b : AlexNet avec 3 convolutions 256x3x3 (padding 1) + ReLU
    ajoutées juste après Conv5 (avant le dernier MaxPool)."""

    def __init__(self, num_classes=1000):
        super().__init__(num_classes)
        layers = list(self.features.children())
        base = layers[:-1]       # tout sauf le dernier MaxPool
        last_pool = layers[-1]   # MaxPool3
        extra = []
        for _ in range(3):
            extra += [nn.Conv2d(256, 256, kernel_size=3, padding=1),
                      nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*base, *extra, last_pool)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Exercice 3 : évaluation, entraînement, Early Stopping


@torch.no_grad()
def evaluate(loader, model, criterion):
    model.eval()  # Désactive le Dropout !
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def build_model(model_name):
    if model_name == "alexnet":
        return AlexNet(num_classes=1000)
    if model_name == "alexnet_deep":
        return AlexNetDeep(num_classes=1000)
    if model_name == "mlp":
        return MLP(num_classes=1000)
    raise ValueError("Modèle inconnu")


def train_model(model_name="alexnet", batch_size=256, lr=1e-4, num_epochs=20,
                weight_decay=1e-5, patience=5, subset_size=20000):
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, subset_size=subset_size)

    model = build_model(model_name).to(device)
    print(f"Modèle {model_name} initialisé avec {count_parameters(model):,} paramètres.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    run_tag = f"{model_name}_sub{subset_size}_lr{lr}"
    writer = SummaryWriter(log_dir=f"runs/{run_tag}")

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()  # Active le Dropout !
        running_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        val_loss, val_acc = evaluate(val_loader, model, criterion)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.flush()

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} (Acc: {train_acc:.4f}) "
              f"| Val Loss: {val_loss:.4f} (Acc: {val_acc:.4f})")

        # --- EARLY STOPPING ---
        if val_loss < best_val_loss:
            # amélioration : on garde le meilleur modèle et on remet le compteur à 0
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_{run_tag}.pt")
        else:
            # pas d'amélioration : on compte
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping déclenché !")
                break

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="alexnet", choices=["mlp", "alexnet", "alexnet_deep"])
    parser.add_argument("--subset", type=int, default=None,
                        help="taille du sous-ensemble (défaut : ImageNet complet)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--params", action="store_true",
                        help="affiche seulement le nombre de paramètres puis quitte")
    args = parser.parse_args()

    if args.params:
        for name in ["mlp", "alexnet", "alexnet_deep"]:
            print(f"{name:13s}: {count_parameters(build_model(name)):,} paramètres")
    else:
        # Sans arguments : entraînement long (Exercice 4) = AlexNet, ImageNet complet, 50 époques
        train_model(model_name=args.model, subset_size=args.subset, num_epochs=args.epochs)
