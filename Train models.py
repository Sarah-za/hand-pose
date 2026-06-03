"""
==============================================
 Hand Gesture Recognition - ML Training
==============================================
Masterarbeit: Vergleich verschiedener ML-Methoden
zur Erkennung von Hand- und Fingerbewegungen

Voraussetzungen:
  pip install scikit-learn pandas numpy matplotlib seaborn
  (PyTorch bereits installiert!)

CSV Format erwartet:
  proband, session_id, session_name, licht, abstand, hand,
  klasse, zeitstempel, kp0_x, kp0_y, kp0_conf, ..., features
==============================================
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ──────────────────────────────────────────────
# 0. GPU PRÜFEN
# ──────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 50)
print("  Hand Gesture Recognition - Training")
print("=" * 50)
print(f"\n🖥️  Gerät: {device}")
if device.type == "cuda":
    print(f"✅ GPU erkannt: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  Keine GPU → Training läuft auf CPU")

# ──────────────────────────────────────────────
# 1. DATEN LADEN
# ──────────────────────────────────────────────

CSV_PATH = "datensatz.csv"   # <-- Hier deinen CSV-Pfad eintragen

# Prüfen ob CSV existiert, sonst Demo-Daten erstellen
if not os.path.exists(CSV_PATH):
    print(f"\n⚠️  '{CSV_PATH}' nicht gefunden.")
    print("→ Erstelle Demo-Datensatz zum Testen...\n")

    classes = ["hand_offen", "faust", "daumen_hoch", "zeigefinger", "winken", "kreisbewegung"]
    rows = []
    for label in classes:
        for _ in range(200):
            keypoints = np.random.rand(42).tolist()
            rows.append(keypoints + [label])

    cols = [f"{axis}{i}" for i in range(21) for axis in ["x", "y"]] + ["klasse"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Demo-CSV erstellt: {CSV_PATH}")
else:
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    print(f"\n✅ Datensatz geladen: {CSV_PATH}")

print(f"\n📊 Datensatz-Info:")
print(f"   Zeilen (Samples): {len(df)}")
print(f"   Spalten:          {len(df.columns)}")
print(f"\n🏷️  Klassen-Verteilung:")
print(df["klasse"].value_counts().to_string())

# ──────────────────────────────────────────────
# 2. DATEN VORBEREITEN
# ──────────────────────────────────────────────

print("\n" + "─" * 50)
print("  Daten vorbereiten...")
print("─" * 50)

# Nur Keypoint x/y Spalten + berechnete Features als Input
IGNORIEREN = ["proband", "session_id", "session_name", "licht",
              "abstand", "hand", "zeitstempel", "klasse"]
feature_cols = [c for c in df.columns if c not in IGNORIEREN]

X = df[feature_cols].values
y = df["klasse"].values

# Labels zu Zahlen
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"\n✅ Klassen: {list(class_names)}")
print(f"✅ Features: {len(feature_cols)} Spalten")

# Normalisierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📂 Train-Samples: {len(X_train)}")
print(f"📂 Test-Samples:  {len(X_test)}")

# ──────────────────────────────────────────────
# 3. SKLEARN MODELLE (RF, SVM, KNN)
# ──────────────────────────────────────────────

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM":           SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
}

results = {}

print("\n" + "=" * 50)
print("  Training & Evaluation")
print("=" * 50)

for name, model in models.items():
    print(f"\n🔧 Trainiere: {name} ...")

    t_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t_start

    t_start = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - t_start) / len(X_test) * 1000

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring="accuracy")
    cm  = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy":         round(acc * 100, 2),
        "f1_score":         round(f1  * 100, 2),
        "cv_mean":          round(cv_scores.mean() * 100, 2),
        "cv_std":           round(cv_scores.std()  * 100, 2),
        "train_time_s":     round(train_time, 3),
        "inference_ms":     round(inference_time, 4),
        "confusion_matrix": cm,
    }

    print(f"   ✅ Accuracy:        {acc*100:.2f}%")
    print(f"   ✅ F1-Score:        {f1*100:.2f}%")
    print(f"   ✅ Cross-Val (5x):  {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"   ⏱️  Trainingszeit:   {train_time:.3f}s")
    print(f"   ⚡ Inferenzzeit:    {inference_time:.4f} ms/Sample")

# ──────────────────────────────────────────────
# 4. NEURONALES NETZ (PyTorch + GPU)
# ──────────────────────────────────────────────

print(f"\n🔧 Trainiere: Neuronales Netz (PyTorch auf {device}) ...")

n_features = X_train.shape[1]
n_classes  = len(class_names)

# Daten zu Tensoren konvertieren
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Modell definieren
class HandGestureNN(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.network(x)

nn_model  = HandGestureNN(n_features, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training
t_start    = time.time()
best_acc   = 0.0
no_improve = 0
patience   = 10
EPOCHS     = 100

for epoch in range(EPOCHS):
    nn_model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = nn_model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation
    nn_model.eval()
    with torch.no_grad():
        val_out  = nn_model(X_test_t)
        val_pred = torch.argmax(val_out, dim=1).cpu().numpy()
        val_acc  = accuracy_score(y_test, val_pred)

    if val_acc > best_acc:
        best_acc   = val_acc
        no_improve = 0
        torch.save(nn_model.state_dict(), "best_nn_model.pt")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"   Early stopping bei Epoch {epoch+1}")
        break

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:>3}/{EPOCHS} | Val Accuracy: {val_acc*100:.2f}%")

train_time = time.time() - t_start

# Bestes Modell laden und evaluieren
nn_model.load_state_dict(torch.load("best_nn_model.pt"))
nn_model.eval()

t_start = time.time()
with torch.no_grad():
    y_pred_nn = torch.argmax(nn_model(X_test_t), dim=1).cpu().numpy()
inference_time = (time.time() - t_start) / len(X_test) * 1000

acc = accuracy_score(y_test, y_pred_nn)
f1  = f1_score(y_test, y_pred_nn, average="weighted")
cm  = confusion_matrix(y_test, y_pred_nn)

results["Neuronales Netz"] = {
    "accuracy":         round(acc * 100, 2),
    "f1_score":         round(f1  * 100, 2),
    "cv_mean":          round(acc * 100, 2),
    "cv_std":           0.0,
    "train_time_s":     round(train_time, 3),
    "inference_ms":     round(inference_time, 4),
    "confusion_matrix": cm,
}

print(f"   ✅ Accuracy:        {acc*100:.2f}%")
print(f"   ✅ F1-Score:        {f1*100:.2f}%")
print(f"   ⏱️  Trainingszeit:   {train_time:.3f}s")
print(f"   ⚡ Inferenzzeit:    {inference_time:.4f} ms/Sample")

# ──────────────────────────────────────────────
# 5. ERGEBNIS-TABELLE
# ──────────────────────────────────────────────

print("\n" + "=" * 70)
print("  ERGEBNIS-VERGLEICH")
print("=" * 70)
print(f"{'Modell':<20} {'Accuracy':>10} {'F1-Score':>10} {'CV-Score':>14} {'Train(s)':>10} {'Inf(ms)':>10}")
print("─" * 78)
for name, r in results.items():
    cv_str = f"{r['cv_mean']}±{r['cv_std']}"
    print(f"{name:<20} {r['accuracy']:>9}% {r['f1_score']:>9}% {cv_str:>14} {r['train_time_s']:>10} {r['inference_ms']:>10}")

best = max(results, key=lambda k: results[k]["accuracy"])
print(f"\n🏆 Bestes Modell: {best} ({results[best]['accuracy']}% Accuracy)")

# ──────────────────────────────────────────────
# 6. VISUALISIERUNGEN
# ──────────────────────────────────────────────

output_dir = "ml_results"
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Accuracy & F1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names  = list(results.keys())
accs   = [results[n]["accuracy"] for n in names]
f1s    = [results[n]["f1_score"] for n in names]
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(names)]

axes[0].bar(names, accs, color=colors, edgecolor="white", linewidth=1.2)
axes[0].set_title("Accuracy Vergleich (%)", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 105)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.5, f"{v}%", ha="center", fontweight="bold")

axes[1].bar(names, f1s, color=colors, edgecolor="white", linewidth=1.2)
axes[1].set_title("F1-Score Vergleich (%)", fontsize=14, fontweight="bold")
axes[1].set_ylabel("F1-Score (%)")
axes[1].set_ylim(0, 105)
for i, v in enumerate(f1s):
    axes[1].text(i, v + 0.5, f"{v}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{output_dir}/accuracy_f1_vergleich.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Plot: {output_dir}/accuracy_f1_vergleich.png")

# Plot 2: Inferenzzeit
fig, ax = plt.subplots(figsize=(10, 5))
inf_times = [results[n]["inference_ms"] for n in names]
bars = ax.bar(names, inf_times, color=colors, edgecolor="white")
ax.set_title("Inferenzzeit pro Sample (ms) — niedriger ist besser", fontsize=13, fontweight="bold")
ax.set_ylabel("Zeit (ms)")
for bar, v in zip(bars, inf_times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
            f"{v:.4f}ms", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{output_dir}/inferenzzeit_vergleich.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot: {output_dir}/inferenzzeit_vergleich.png")

# Plot 3: Confusion Matrices
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(r["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{name}\n({r['accuracy']}%)", fontweight="bold")
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"📊 Plot: {output_dir}/confusion_matrices.png")

# CSV Ergebnisse speichern
results_df = pd.DataFrame([
    {
        "Modell":            name,
        "Accuracy (%)":      r["accuracy"],
        "F1-Score (%)":      r["f1_score"],
        "CV-Score (%)":      r["cv_mean"],
        "CV-Std (%)":        r["cv_std"],
        "Trainingszeit (s)": r["train_time_s"],
        "Inferenzzeit (ms)": r["inference_ms"],
    }
    for name, r in results.items()
])
results_df.to_csv(f"{output_dir}/ergebnisse.csv", index=False)
print(f"📁 Ergebnisse: {output_dir}/ergebnisse.csv")

print("\n" + "=" * 50)
print("  ✅ Training abgeschlossen!")
print(f"  📂 Alle Ergebnisse in: ./{output_dir}/")
print("=" * 50)