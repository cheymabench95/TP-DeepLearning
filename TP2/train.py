import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import CardioDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ============================================================
# 1. Chargement du dataset
# ============================================================

dataset = CardioDataset("data/cardio_train.csv")

print(f"Taille totale du dataset : {len(dataset)}")


# ============================================================
# 2. Découpage Train / Validation / Test
# ============================================================

generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    dataset,
    [0.8, 0.1, 0.1],
    generator=generator
)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)


# ============================================================
# 3. Architecture du MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Choix du CPU / GPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device utilisé : {device}")


# ============================================================
# 5. Détermination automatique du nombre de features
# ============================================================

batch = next(iter(train_loader))

input_size = batch["features"].shape[1]

print(f"Nombre de features : {input_size}")


# ============================================================
# 6. Fonction d'entraînement
# ============================================================

def train_model(opt_name, learning_rate=0.001, epochs=30):

    print("\n==========================================")
    print(f"Entraînement avec {opt_name}")
    print("==========================================")

    # Nouveau modèle pour chaque optimiseur
    model = MLP(
        input_size=input_size,
        hidden_size=128
    ).to(device)

    # Fonction de coût
    criterion = nn.BCELoss()

    # ========================================================
    # Choix de l'optimiseur
    # ========================================================

    if opt_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:
        raise ValueError(
            f"Optimiseur inconnu : {opt_name}"
        )

    # ========================================================
    # TensorBoard
    # ========================================================

    writer = SummaryWriter(
        f"runs/cardio_{opt_name}_lr{learning_rate}"
    )

    # ========================================================
    # Entraînement
    # ========================================================

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            # Remise à zéro des gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)

            # Calcul de la loss
            loss = criterion(outputs, targets)

            # Backpropagation
            loss.backward()

            # Mise à jour des poids
            optimizer.step()

            # Somme des losses
            running_loss += loss.item()

        # Loss moyenne de l'epoch
        average_loss = running_loss / len(train_loader)

        print(
            f"{opt_name} - "
            f"Epoch [{epoch + 1}/{epochs}] "
            f"- Loss: {average_loss:.4f}"
        )

        # Enregistrement dans TensorBoard
        writer.add_scalar(
            "Training Loss",
            average_loss,
            epoch
        )

    writer.close()

    return model

# ============================================================
# 7. Fonction d'évaluation
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    all_targets = []
    all_preds_probs = []

    # Désactivation du calcul des gradients
    with torch.no_grad():

        for batch in test_loader:

            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(inputs)

            # Stockage des cibles et probabilités
            all_targets.extend(
                targets.cpu().numpy()
            )

            all_preds_probs.extend(
                outputs.cpu().numpy()
            )

    # Conversion en tableaux NumPy
    all_targets = np.array(all_targets)
    all_preds_probs = np.array(all_preds_probs)

    # Prédictions binaires avec un seuil de 0.5
    all_preds_classes = (
        all_preds_probs > 0.5
    ).astype(int)

    # Calcul des métriques
    precision = precision_score(
        all_targets,
        all_preds_classes
    )

    recall = recall_score(
        all_targets,
        all_preds_classes
    )

    f1 = f1_score(
        all_targets,
        all_preds_classes
    )

    auc = roc_auc_score(
        all_targets,
        all_preds_probs
    )

    print("\n==========================================")
    print("Résultats sur l'ensemble de test")
    print("==========================================")

    print(
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"AUC: {auc:.4f}"
    )

# ============================================================
# 8. Entraînement du modèle à évaluer
# ============================================================

model = train_model(
    "RMSprop",
    learning_rate=0.001,
    epochs=30
)


# ============================================================
# 9. Évaluation sur l'ensemble de test
# ============================================================

evaluate_model(
    model,
    test_loader
)
