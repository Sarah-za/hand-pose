"""
==============================================
 Hand Gesture Recognition - ML Training
==============================================
Masterarbeit: Vergleich verschiedener ML-Methoden
zur Erkennung von Hand- und Fingerbewegungen

Voraussetzungen (installieren):
  pip install scikit-learn pandas numpy matplotlib seaborn tensorflow

CSV Format erwartet:
  x0,y0,x1,y1,...,x20,y20,label
  (21 Keypoints = 42 Spalten + 1 Label-Spalte)
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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ──────────────────────────────────────────────
# 1. DATEN LADEN
# ──────────────────────────────────────────────

CSV_PATH = r"C:\Users\Latitude 5510\Desktop\Masterarbeit\hand-pose\datensatz.csv"   # <-- Hier CSV-Pfad eintragen

print("=" * 50)
print("  Handgesten erkennen- Training")
print("=" * 50)

# Prüfen ob CSV existiert, sonst Demo-Daten erstellen
if not os.path.exists(CSV_PATH):
    print(f"\n⚠️  '{CSV_PATH}' nicht gefunden.")
    print("→ Erstelle Demo-Datensatz zum Testen...\n")

    # Demo-Daten: 6 Klassen, 200 Samples pro Klasse
    classes = ["faust", "hand_offen", "daumen_hoch", "zeigefinger", "winken", "kreisbewegung"]
    rows = []
    for label in classes:
        for _ in range(200):
            keypoints = np.random.rand(42).tolist()  # 21 x/y Koordinaten
            rows.append(keypoints + [label])

    cols = [f"{axis}{i}" for i in range(21) for axis in ["x", "y"]] + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Demo-CSV erstellt: {CSV_PATH}")
else:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅ Datensatz geladen: {CSV_PATH}")

print(f"\n📊 Datensatz-Info:")
print(f"   Zeilen (Samples): {len(df)}")
print(f"   Spalten:          {len(df.columns)}")
print(f"\n🏷️  Klassen-Verteilung:")
print(df["label"].value_counts().to_string())

# ──────────────────────────────────────────────
# 2. DATEN VORBEREITEN
# ──────────────────────────────────────────────

print("\n" + "─" * 50)
print("  Daten vorbereiten...")
print("─" * 50)

# Features (X) und Labels (y) trennen
X = df.drop(columns=["label"]).values
y = df["label"].values

# Labels zu Zahlen umwandeln
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"\n✅ Klassen erkannt: {list(class_names)}")

# Normalisierung (wichtig für SVM und KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test Split (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📂 Train-Samples: {len(X_train)}")
print(f"📂 Test-Samples:  {len(X_test)}")

# ──────────────────────────────────────────────
# 3. MODELLE DEFINIEREN
# ──────────────────────────────────────────────

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SVM":           SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "KNN":           KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
}

# ──────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ──────────────────────────────────────────────

results = {}

print("\n" + "=" * 50)
print("  Training & Evaluation")
print("=" * 50)

for name, model in models.items():
    print(f"\n🔧 Trainiere: {name} ...")

    # Training
    t_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t_start

    # Inference (Vorhersage auf Testdaten)
    t_start = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - t_start) / len(X_test) * 1000  # ms pro Sample

    # Metriken
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring="accuracy")
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy":       round(acc * 100, 2),
        "f1_score":       round(f1 * 100, 2),
        "cv_mean":        round(cv_scores.mean() * 100, 2),
        "cv_std":         round(cv_scores.std() * 100, 2),
        "train_time_s":   round(train_time, 3),
        "inference_ms":   round(inference_time, 4),
        "confusion_matrix": cm,
        "y_pred":         y_pred,
    }

    print(f"   ✅ Accuracy:        {acc*100:.2f}%")
    print(f"   ✅ F1-Score:        {f1*100:.2f}%")
    print(f"   ✅ Cross-Val (5x):  {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"   ⏱️  Trainingszeit:   {train_time:.3f}s")
    print(f"   ⚡ Inferenzzeit:    {inference_time:.4f} ms/Sample")

# ──────────────────────────────────────────────
# 5. NEURONALES NETZ (TensorFlow)
# ──────────────────────────────────────────────

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    print(f"\n🔧 Trainiere: Neuronales Netz ...")

    n_classes = len(class_names)
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

    nn_model = Sequential([
        Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])

    nn_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(patience=10, restore_best_weights=True, verbose=0)

    t_start = time.time()
    history = nn_model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )
    train_time = time.time() - t_start

    t_start = time.time()
    y_pred_prob = nn_model.predict(X_test, verbose=0)
    inference_time = (time.time() - t_start) / len(X_test) * 1000
    y_pred_nn = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred_nn)
    f1  = f1_score(y_test, y_pred_nn, average="weighted")
    cm  = confusion_matrix(y_test, y_pred_nn)

    results["Neuronales Netz"] = {
        "accuracy":       round(acc * 100, 2),
        "f1_score":       round(f1 * 100, 2),
        "cv_mean":        round(acc * 100, 2),
        "cv_std":         0.0,
        "train_time_s":   round(train_time, 3),
        "inference_ms":   round(inference_time, 4),
        "confusion_matrix": cm,
        "y_pred":         y_pred_nn,
        "history":        history,
    }

    print(f"   ✅ Accuracy:        {acc*100:.2f}%")
    print(f"   ✅ F1-Score:        {f1*100:.2f}%")
    print(f"   ⏱️  Trainingszeit:   {train_time:.3f}s")
    print(f"   ⚡ Inferenzzeit:    {inference_time:.4f} ms/Sample")

except ImportError:
    print("\n⚠️  TensorFlow nicht installiert → Neuronales Netz wird übersprungen.")
    print("   Installieren mit: pip install tensorflow")

# ──────────────────────────────────────────────
# 6. ERGEBNIS-TABELLE
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("  ERGEBNIS-VERGLEICH")
print("=" * 60)
print(f"{'Modell':<20} {'Accuracy':>10} {'F1-Score':>10} {'CV-Score':>12} {'Train(s)':>10} {'Inf(ms)':>10}")
print("─" * 75)
for name, r in results.items():
    cv_str = f"{r['cv_mean']}±{r['cv_std']}"
    print(f"{name:<20} {r['accuracy']:>9}% {r['f1_score']:>9}% {cv_str:>12} {r['train_time_s']:>10} {r['inference_ms']:>10}")

# Bestes Modell
best = max(results, key=lambda k: results[k]["accuracy"])
print(f"\n🏆 Bestes Modell: {best} ({results[best]['accuracy']}% Accuracy)")

# ──────────────────────────────────────────────
# 7. VISUALISIERUNGEN
# ──────────────────────────────────────────────

output_dir = "ml_results"
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Accuracy & F1 Vergleich
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names   = list(results.keys())
accs    = [results[n]["accuracy"] for n in names]
f1s     = [results[n]["f1_score"] for n in names]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(names)]

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
print(f"\n📊 Plot gespeichert: {output_dir}/accuracy_f1_vergleich.png")

# Plot 2: Inferenzzeit Vergleich
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
print(f"📊 Plot gespeichert: {output_dir}/inferenzzeit_vergleich.png")

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
print(f"📊 Plot gespeichert: {output_dir}/confusion_matrices.png")

# Plot 4: Neuronales Netz Lernkurve (falls vorhanden)
if "Neuronales Netz" in results and "history" in results["Neuronales Netz"]:
    history = results["Neuronales Netz"]["history"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history["accuracy"],     label="Train Accuracy", linewidth=2)
    ax.plot(history.history["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
    ax.set_title("Neuronales Netz - Lernkurve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lernkurve_nn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Plot gespeichert: {output_dir}/lernkurve_nn.png")

# ──────────────────────────────────────────────
# 8. CSV ERGEBNIS-TABELLE SPEICHERN
# ──────────────────────────────────────────────

results_df = pd.DataFrame([
    {
        "Modell":          name,
        "Accuracy (%)":    r["accuracy"],
        "F1-Score (%)":    r["f1_score"],
        "CV-Score (%)":    r["cv_mean"],
        "CV-Std (%)":      r["cv_std"],
        "Trainingszeit (s)": r["train_time_s"],
        "Inferenzzeit (ms)": r["inference_ms"],
    }
    for name, r in results.items()
])

results_df.to_csv(f"{output_dir}/ergebnisse.csv", index=False)
print(f"📁 Ergebnisse gespeichert: {output_dir}/ergebnisse.csv")

print("\n" + "=" * 50)
print("  ✅ Training abgeschlossen!")
print(f"  📂 Alle Ergebnisse in: ./{output_dir}/")
print("=" * 50)