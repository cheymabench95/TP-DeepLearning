
import os
import json
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# CONFIGURATION GENERALE
# ============================================================

SEED = 42
DATA_DIR = "/mnt/ssd/irishman"
SEQ_LENGTH = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROMPT = "X:1\nT:Irish Reel\nM:4/4\nL:1/8\nK:D\n"
TEMPERATURES = [0.2, 0.8, 2.0]


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# EXERCICE 1 : CHARGEMENT ET ANALYSE DES DONNEES
# ============================================================

def load_json(filename):
    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_json("train.json")
val_data = load_json("validation.json")

train_songs = [item["abc notation"] for item in train_data]
val_songs = [item["abc notation"] for item in val_data]

all_songs = train_songs + val_songs

unique_chars = sorted(
    set("".join(all_songs))
)

char_to_index = {
    char: index
    for index, char in enumerate(unique_chars)
}

index_to_char = {
    index: char
    for char, index in char_to_index.items()
}

vocab_size = len(unique_chars)

lengths = [len(song) for song in all_songs]

print(f"Nombre de chansons (Train) : {len(train_songs)}")
print(f"Nombre de chansons (Val)   : {len(val_songs)}")

print("\n--- Exemple de partition (Train[0]) ---")
print(train_songs[0])

print(f"\nTaille du vocabulaire : {vocab_size} caracteres uniques")
print(f"Longueur moyenne : {sum(lengths) / len(lengths):.1f}")
print(f"Longueur maximale : {max(lengths)}")
print(f"Appareil utilise : {DEVICE}")


# ============================================================
# EXERCICE 2 : DATASET
# ============================================================

class MusicDataset(Dataset):

    def __init__(self, songs, subset_size=None):
        if subset_size is not None:
            songs = songs[:subset_size]

        self.songs = songs
        self.pad_index = char_to_index[" "]

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, index):
        song = self.songs[index]

        indices = [
            char_to_index[char]
            for char in song[:SEQ_LENGTH]
        ]

        if len(indices) < SEQ_LENGTH:
            indices += [
                self.pad_index
            ] * (SEQ_LENGTH - len(indices))

        sequence = torch.tensor(
            indices,
            dtype=torch.long
        )

        x = sequence[:-1]
        y = sequence[1:]

        return x, y


# ============================================================
# EXERCICE 2 : MODELES RNN ET LSTM
# ============================================================

class MusicModel(nn.Module):

    def __init__(
        self,
        rnn_type="lstm",
        vocab_size=95,
        embedding_dim=256,
        hidden_size=512
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True
            )

        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True
            )

        else:
            raise ValueError(
                f"Type de reseau inconnu : {rnn_type}"
            )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x, hidden_state=None):
        embedded = self.embedding(x)

        output, hidden_state = self.rnn(
            embedded,
            hidden_state
        )

        logits = self.fc(output)

        return logits, hidden_state


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def check_exercise_2():
    dataset = MusicDataset(
        train_songs,
        subset_size=128
    )

    x, y = dataset[0]

    print("\n--- Verification du Dataset ---")
    print("Shape de x :", tuple(x.shape))
    print("Shape de y :", tuple(y.shape))
    print("Type de x :", x.dtype)
    print("Type de y :", y.dtype)

    batch = next(iter(
        DataLoader(dataset, batch_size=128)
    ))

    batch_x, batch_y = batch

    print("Shape batch x :", tuple(batch_x.shape))
    print("Shape batch y :", tuple(batch_y.shape))

    for rnn_type in ["rnn", "lstm"]:
        model = MusicModel(
            rnn_type=rnn_type,
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_size=512
        ).to(DEVICE)

        print(
            f"{rnn_type.upper()} : "
            f"{count_parameters(model)} parametres"
        )

        with torch.no_grad():
            logits, _ = model(
                batch_x[:2].to(DEVICE)
            )

        print(
            "Shape logits :",
            tuple(logits.shape)
        )


# ============================================================
# EXERCICE 3 : EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(model, data_loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in data_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        predictions = logits.argmax(dim=-1)

        number_tokens = y.numel()

        total_loss += loss.item() * number_tokens

        total_correct += (
            predictions == y
        ).sum().item()

        total_tokens += number_tokens

    average_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    return average_loss, accuracy


# ============================================================
# EXERCICE 3 : ENTRAINEMENT
# ============================================================

def train_model(
    rnn_type="lstm",
    subset_size=10000,
    num_epochs=15,
    batch_size=128,
    lr=5e-3,
    embedding_dim=256,
    hidden_size=512,
    max_grad_norm=1.0,
    patience=5,
    run_name="lstm_clip"
):

    print("\n" + "=" * 65)
    print(f"Experience : {run_name}")
    print(f"Reseau : {rnn_type.upper()}")
    print(f"Nombre de chansons : {subset_size or len(train_songs)}")
    print(f"Epochs : {num_epochs}")
    print(f"Batch size : {batch_size}")
    print(f"Learning rate : {lr}")
    print(f"Hidden size : {hidden_size}")
    print(f"Gradient clipping : {max_grad_norm}")
    print("=" * 65)

    train_dataset = MusicDataset(
        train_songs,
        subset_size=subset_size
    )

    val_dataset = MusicDataset(val_songs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = MusicModel(
        rnn_type=rnn_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size
    ).to(DEVICE)

    print(
        f"Nombre de parametres : "
        f"{count_parameters(model):,}"
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    writer = SummaryWriter(
        log_dir=os.path.join(
            "runs_rnn",
            run_name
        )
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    checkpoint_path = f"best_{run_name}.pth"

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_train_loss = 0.0
        total_train_tokens = 0

        total_grad_norm = 0.0
        number_batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            loss.backward()

            if max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                ).item()
            else:
                grad_norm_squared = 0.0

                for parameter in model.parameters():
                    if parameter.grad is not None:
                        norm = parameter.grad.detach().norm(2)
                        grad_norm_squared += norm.item() ** 2

                grad_norm = grad_norm_squared ** 0.5

            optimizer.step()

            number_tokens = y.numel()

            total_train_loss += (
                loss.item() * number_tokens
            )

            total_train_tokens += number_tokens

            total_grad_norm += grad_norm
            number_batches += 1

            writer.add_scalar(
                "GradNorm/step",
                grad_norm,
                global_step
            )

            global_step += 1

        train_loss = (
            total_train_loss / total_train_tokens
        )

        avg_grad_norm = (
            total_grad_norm / number_batches
        )

        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion
        )

        writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/Validation",
            val_accuracy,
            epoch
        )

        writer.add_scalar(
            "GradNorm/Epoch",
            avg_grad_norm,
            epoch
        )

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Grad Norm: {avg_grad_norm:.4f}",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "rnn_type": rnn_type,
                    "vocab_size": vocab_size,
                    "embedding_dim": embedding_dim,
                    "hidden_size": hidden_size,
                    "char_to_index": char_to_index,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
                checkpoint_path
            )

            print(
                f"  -> Meilleur modele sauvegarde : "
                f"{checkpoint_path}",
                flush=True
            )

        else:
            epochs_without_improvement += 1

        if (
            patience is not None
            and epochs_without_improvement >= patience
        ):
            print(
                f"Early stopping a l'epoch {epoch}",
                flush=True
            )
            break

    duration = (time.time() - start_time) / 60

    writer.close()

    print("\n--- Fin de l'entrainement ---")
    print(f"Meilleure Val Loss : {best_val_loss:.4f}")
    print(f"Duree : {duration:.1f} minutes")
    print(f"Checkpoint : {checkpoint_path}")

    return checkpoint_path


# ============================================================
# EXERCICE 4 : GENERATION AVEC ETAT CACHE
# ============================================================

@torch.no_grad()
def generate_music(
    model,
    start_sequence,
    output_length=300,
    temperature=1.0
):

    if temperature <= 0:
        raise ValueError(
            "La temperature doit etre strictement positive."
        )

    model.eval()

    generated_sequence = start_sequence

    input_indices = [
        char_to_index[char]
        for char in start_sequence
    ]

    input_tensor = torch.tensor(
        [input_indices],
        dtype=torch.long,
        device=DEVICE
    )

    # Lecture de l'amorce une seule fois.
    logits, hidden_state = model(input_tensor)

    # On reutilise l'etat cache a chaque caractere.
    for _ in range(output_length):

        last_logits = (
            logits[:, -1, :] / temperature
        )

        probabilities = F.softmax(
            last_logits,
            dim=-1
        )

        next_index = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        next_character = index_to_char[next_index]

        generated_sequence += next_character

        next_input = torch.tensor(
            [[next_index]],
            dtype=torch.long,
            device=DEVICE
        )

        logits, hidden_state = model(
            next_input,
            hidden_state
        )

    return generated_sequence


# ============================================================
# EXERCICE 4 : CHARGEMENT D'UN CHECKPOINT
# ============================================================

def load_trained_model(checkpoint_path):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(DEVICE)
        model.eval()
        return model

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Format de checkpoint non reconnu."
        )

    state_dict = None

    for key in [
        "model_state_dict",
        "model_state",
        "state_dict"
    ]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        state_dict = checkpoint

    # Compatibilite avec differents noms d'attributs.
    normalized_state = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        for old_prefix, new_prefix in [
            ("lstm.", "rnn."),
            ("recurrent.", "rnn."),
            ("output.", "fc."),
            ("linear.", "fc."),
            ("classifier.", "fc."),
            ("embed.", "embedding.")
        ]:
            if new_key.startswith(old_prefix):
                new_key = (
                    new_prefix
                    + new_key[len(old_prefix):]
                )
                break

        normalized_state[new_key] = value

    required_keys = [
        "embedding.weight",
        "rnn.weight_ih_l0",
        "rnn.weight_hh_l0",
        "fc.weight"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in normalized_state
    ]

    if missing_keys:
        raise ValueError(
            "Checkpoint incompatible. Cles manquantes : "
            + str(missing_keys)
            + "\nCles disponibles : "
            + str(list(state_dict.keys())[:20])
        )

    checkpoint_vocab_size = (
        normalized_state["embedding.weight"].shape[0]
    )

    embedding_dim = (
        normalized_state["embedding.weight"].shape[1]
    )

    hidden_size = (
        normalized_state["rnn.weight_hh_l0"].shape[1]
    )

    gate_size = (
        normalized_state["rnn.weight_ih_l0"].shape[0]
    )

    if gate_size == 4 * hidden_size:
        rnn_type = "lstm"
    elif gate_size == hidden_size:
        rnn_type = "rnn"
    else:
        raise ValueError(
            "Impossible de determiner le type de reseau."
        )

    if checkpoint_vocab_size != vocab_size:
        raise ValueError(
            f"Vocabulaire incompatible : "
            f"checkpoint={checkpoint_vocab_size}, "
            f"actuel={vocab_size}"
        )

    saved_mapping = checkpoint.get("char_to_index")

    if saved_mapping is not None:
        if saved_mapping != char_to_index:
            raise ValueError(
                "Le vocabulaire du checkpoint ne correspond "
                "pas au vocabulaire actuel."
            )

    model = MusicModel(
        rnn_type=rnn_type,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size
    ).to(DEVICE)

    model.load_state_dict(normalized_state)
    model.eval()

    print("\n--- Modele charge ---")
    print("Checkpoint :", checkpoint_path)
    print("Type :", rnn_type.upper())
    print("Embedding :", embedding_dim)
    print("Hidden size :", hidden_size)

    return model


# ============================================================
# EXERCICE 4 : TROIS TEMPERATURES
# ============================================================

def generate_three_examples(
    checkpoint_path="best_lstm_clip.pth",
    output_dir="generated_music"
):

    os.makedirs(output_dir, exist_ok=True)

    model = load_trained_model(
        checkpoint_path
    )

    print("\n--- Generation de partitions ABC ---")
    print("Amorce :")
    print(PROMPT)

    for temperature in TEMPERATURES:

        music = generate_music(
            model=model,
            start_sequence=PROMPT,
            output_length=300,
            temperature=temperature
        )

        filename = (
            "music_T_"
            + str(temperature).replace(".", "_")
            + ".abc"
        )

        filepath = os.path.join(
            output_dir,
            filename
        )

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(music)

        print("\n" + "=" * 65)
        print(f"TEMPERATURE = {temperature}")
        print(f"FICHIER = {filepath}")
        print("=" * 65)
        print(music)
        print("=" * 65, flush=True)

    print(
        "\nLes trois partitions ont ete generees.",
        flush=True
    )


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "TP4 - RNN, LSTM et generation "
            "de musique ABC"
        )
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "check",
            "rnn_no_clip",
            "rnn_clip",
            "lstm_clip",
            "generate",
            "lstm_final"
        ],
        default="check",
        help="Experience a executer"
    )

    args = parser.parse_args()

    if args.experiment == "check":

        check_exercise_2()

    elif args.experiment == "rnn_no_clip":

        train_model(
            rnn_type="rnn",
            subset_size=10000,
            num_epochs=15,
            batch_size=128,
            lr=5e-3,
            embedding_dim=256,
            hidden_size=512,
            max_grad_norm=None,
            patience=5,
            run_name="rnn_no_clip"
        )

    elif args.experiment == "rnn_clip":

        train_model(
            rnn_type="rnn",
            subset_size=10000,
            num_epochs=15,
            batch_size=128,
            lr=5e-3,
            embedding_dim=256,
            hidden_size=512,
            max_grad_norm=1.0,
            patience=5,
            run_name="rnn_clip"
        )

    elif args.experiment == "lstm_clip":

        train_model(
            rnn_type="lstm",
            subset_size=10000,
            num_epochs=15,
            batch_size=128,
            lr=5e-3,
            embedding_dim=256,
            hidden_size=512,
            max_grad_norm=1.0,
            patience=5,
            run_name="lstm_clip"
        )

    elif args.experiment == "generate":

        generate_three_examples(
            checkpoint_path="best_lstm_clip.pth",
            output_dir="generated_music"
        )

    elif args.experiment == "lstm_final":

        checkpoint_path = train_model(
            rnn_type="lstm",
            subset_size=50000,
            num_epochs=30,
            batch_size=64,
            lr=1e-3,
            embedding_dim=256,
            hidden_size=1024,
            max_grad_norm=1.0,
            patience=None,
            run_name="lstm_final"
        )

        generate_three_examples(
            checkpoint_path=checkpoint_path,
            output_dir="generated_music_final"
        )
