from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Dossiers
log_dir = Path("runs/captioning")
fig_dir = Path("figures")
fig_dir.mkdir(exist_ok=True)

# Sélectionner le fichier TensorBoard le plus récent :
# celui de l'entraînement complet de 15 époques.
event_files = list(log_dir.glob("events.out.tfevents.*"))

if not event_files:
    raise FileNotFoundError("Aucun fichier TensorBoard trouvé.")

event_file = max(event_files, key=lambda p: p.stat().st_mtime)

print("Fichier utilisé :", event_file.name)

# Lire toutes les valeurs enregistrées
acc = EventAccumulator(
    str(event_file),
    size_guidance={"scalars": 0}
)
acc.Reload()

tags = acc.Tags().get("scalars", [])

print("Courbes disponibles :", tags)

if not tags:
    raise RuntimeError("Aucune donnée scalaire trouvée.")

# Créer une figure PNG pour chaque courbe
for tag in tags:
    events = acc.Scalars(tag)

    if not events:
        continue

    # Les courbes enregistrées par époque commencent à 0
    by_epoch = tag in ("Loss/Train", "Loss/Validation", "LR")

    x = [
        e.step + 1 if by_epoch else e.step
        for e in events
    ]
    y = [e.value for e in events]

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, linewidth=1.8)

    plt.title(tag)
    plt.xlabel("Époque" if by_epoch else "Étape d'entraînement")
    plt.ylabel("Valeur")
    plt.grid(True, alpha=0.3)

    filename = tag.replace("/", "_") + ".png"
    output = fig_dir / filename

    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Figure enregistrée : {output}")

print("\nExport terminé !")
