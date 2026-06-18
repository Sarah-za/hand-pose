import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────────────────────
# 0. KONFIGURATION
#    Hier werden alle wichtigen Einstellungen zentral festgelegt,
#    damit sie leicht geändert werden können ohne den Code zu durchsuchen.
# ──────────────────────────────────────────────────────────────────────────────

DATENSATZ_ORDNER = "datensatz"   # Ordner mit allen rohen CSV-Dateien
MODEL_ORDNER     = "modelle"     # Ordner zum Speichern der Modelle
TEST_SIZE    = 0.2                               # 20% der Daten für den Test
RANDOM_STATE = 42                                # Fixierter Zufallswert → reproduzierbare Ergebnisse

# Namen der 6 Gestenklassen (0–5)
KLASSEN_NAMEN = {
    0: "hand_offen",
    1: "faust",
    2: "daumen_hoch",
    3: "zeigefinger",
    4: "winken",
    5: "kreisbewegung"
}

# Nur diese 10 berechneten Features werden als Input für die Modelle verwendet,
# nicht die rohen Keypoint-Koordinaten (zu viele, redundant)
FEATURE_COLS = [
    "distanz_daumen_zeigefinger",
    "distanz_daumen_mittelfinger",
    "distanz_daumen_kleinfinger",
    "winkel_zeigefinger",
    "winkel_mittelfinger",
    "winkel_ringfinger",
    "winkel_kleinfinger",
    "geschwindigkeit_daumen",
    "geschwindigkeit_zeigefinger",
    "handgroesse"
]

# ──────────────────────────────────────────────────────────────────────────────
# 1. GPU PRÜFEN
#    PyTorch kann auf der GPU (NVIDIA) viel schneller trainieren als auf der CPU.
#    Falls keine GPU verfügbar ist, wird automatisch die CPU verwendet.
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 50)
print("  Hand Gesture Recognition - Training")
print("=" * 50)
print(f"\nGerät: {device}")
if device.type == "cuda":
    print(f"GPU erkannt: {torch.cuda.get_device_name(0)}")
else:
    print("Keine GPU gefunden → Training läuft auf CPU")

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATEN LADEN
#    Der bereinigte Datensatz (nach datenbereinigung.py) wird eingelesen.
#    Nur die definierten Feature-Spalten werden als Input (X) verwendet.
#    Die Spalte "klasse" ist der Output (y), den das Modell lernen soll.
# ──────────────────────────────────────────────────────────────────────────────

def lade_daten():
    # Alle CSV-Dateien aus dem Datensatz-Ordner laden und zusammenführen
    import glob
    dateien = sorted(glob.glob(os.path.join(DATENSATZ_ORDNER, "*.csv")))
    print(f"\n{len(dateien)} CSV-Dateien gefunden:")
    dfs = []
    for datei in dateien:
        df_teil = pd.read_csv(datei)
        print(f"  {os.path.basename(datei)} — {len(df_teil)} Zeilen")
        dfs.append(df_teil)
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nGesamt: {len(df)} Samples")

    X = df[FEATURE_COLS].values   # Feature-Matrix: jede Zeile = ein Sample
    y = df["klasse"].values        # Zielvektor: die zugehörige Gestenklasse
    print(f"Features: {X.shape[1]}")
    print("\nKlassenverteilung:")
    print(df["klasse"].value_counts().to_string())
    return X, y

# ──────────────────────────────────────────────────────────────────────────────
# 3. DATEN VORBEREITEN
#    a) StandardScaler: Normalisiert alle Features auf Mittelwert=0, Std=1.
#       Wichtig für SVM und KNN, da diese auf Abstände basieren und
#       sonst Features mit großen Werten dominieren würden.
#    b) LabelEncoder: Wandelt Klassennamen ("faust") in Zahlen (1) um,
#       da ML-Modelle nur mit Zahlen arbeiten können.
#    c) Train/Test-Split: 80% der Daten zum Training, 20% zum Testen.
#       stratify=y stellt sicher, dass alle Klassen gleich verteilt sind.
# ──────────────────────────────────────────────────────────────────────────────

X, y = lade_daten()

# Normalisierung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Labels zu Zahlen
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(np.unique(y_enc))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_enc          # gleiche Klassenverteilung in Train und Test
)
print(f"\nTrain: {len(X_train)} Samples | Test: {len(X_test)} Samples")

# ──────────────────────────────────────────────────────────────────────────────
# 4. RANDOM FOREST TRAINING
#    Ensemble aus 100 Entscheidungsbäumen.
#    class_weight="balanced" korrigiert das Ungleichgewicht zwischen statischen
#    Gesten (50 Frames) und dynamischen Gesten (100 Frames) automatisch.
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
print("  Random Forest")
print("─" * 50)

rf = RandomForestClassifier(
    n_estimators=100,           # Anzahl der Entscheidungsbäume
    class_weight="balanced",    # Gewichtet Klassen nach ihrer Häufigkeit
    random_state=RANDOM_STATE,
    n_jobs=-1                   # Nutzt alle CPU-Kerne parallel
)
rf.fit(X_train, y_train)
print("Training abgeschlossen")

# ──────────────────────────────────────────────────────────────────────────────
# 5. SVM TRAINING
#    Support Vector Machine mit RBF-Kernel (Radial Basis Function).
#    Der RBF-Kernel eignet sich für nicht-linear trennbare Daten,
#    wie es bei Handgesten typischerweise der Fall ist.
#    C=10: Regularisierungsparameter (höher = weniger Toleranz für Fehler)
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
print("  SVM")
print("─" * 50)

svm = SVC(
    kernel="rbf",               # Nichtlinearer Kernel
    C=10,                       # Regularisierungsparameter
    gamma="scale",              # Kernel-Koeffizient (automatisch skaliert)
    class_weight="balanced",    # Klassenungleichgewicht ausgleichen
    probability=True,           # Ermöglicht Wahrscheinlichkeitsausgabe
    random_state=RANDOM_STATE
)
svm.fit(X_train, y_train)
print("Training abgeschlossen")

# ──────────────────────────────────────────────────────────────────────────────
# 6. KNN TRAINING
#    K-Nearest Neighbors klassifiziert eine neue Geste anhand der
#    5 ähnlichsten Trainingssamples (Nachbarn) im Feature-Raum.
#    Kein eigentliches "Training" — KNN speichert nur die Trainingsdaten
#    und sucht beim Vorhersagen die nächsten Nachbarn.
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
print("  KNN")
print("─" * 50)

knn = KNeighborsClassifier(
    n_neighbors=5,              # Anzahl der Nachbarn
    metric="euclidean"          # Euklidischer Abstand als Ähnlichkeitsmaß
)
knn.fit(X_train, y_train)
print("Training abgeschlossen")

# ──────────────────────────────────────────────────────────────────────────────
# 7. NEURONALES NETZ (PyTorch)
#    Vollverbundenes Netz (Fully Connected) mit zwei versteckten Schichten.
#    Architektur: Input(10) → 128 → 64 → Output(6 Klassen)
#    ReLU:      Aktivierungsfunktion, bringt Nichtlinearität ins Netz
#    Dropout:   Deaktiviert zufällig Neuronen → verhindert Overfitting
#    BatchNorm: Normalisiert Aktivierungen → stabileres Training
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "─" * 50)
print(f"  Neuronales Netz (PyTorch auf {device})")
print("─" * 50)

# Modellarchitektur definieren
class HandGestenNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.netz = nn.Sequential(
            nn.Linear(input_dim, 128),  # Erste versteckte Schicht: 10 → 128 Neuronen
            nn.BatchNorm1d(128),         # Normalisierung der Aktivierungen
            nn.ReLU(),                   # Aktivierungsfunktion
            nn.Dropout(0.3),             # 30% der Neuronen werden zufällig deaktiviert

            nn.Linear(128, 64),          # Zweite versteckte Schicht: 128 → 64 Neuronen
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)   # Ausgabeschicht: 64 → 6 Klassen
        )

    def forward(self, x):
        return self.netz(x)

# Klassengewichte berechnen (wegen Ungleichgewicht statisch vs. dynamisch)
counts = np.bincount(y_train)
weights = 1.0 / counts
weights = weights / weights.sum() * num_classes
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

# Daten in PyTorch-Tensoren umwandeln und auf GPU/CPU laden
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)

# DataLoader: teilt Daten in Mini-Batches auf (je 32 Samples pro Schritt)
dataset = TensorDataset(X_train_t, y_train_t)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

# Modell, Verlustfunktion und Optimierer initialisieren
nn_model  = HandGestenNN(X_train.shape[1], num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)   # Verlustfunktion mit Klassengewichten
optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # Lernrate alle 20 Epochen halbieren

# Trainingsschleife mit Early Stopping
# Early Stopping: Training wird abgebrochen wenn sich die Genauigkeit
# über 10 Epochen nicht mehr verbessert → verhindert Overfitting
os.makedirs(MODEL_ORDNER, exist_ok=True)
EPOCHS     = 100
patience   = 10
best_acc   = 0.0
no_improve = 0

from sklearn.metrics import accuracy_score as sk_acc

for epoch in range(EPOCHS):
    # Training
    nn_model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()                   # Gradienten zurücksetzen
        output = nn_model(X_batch)              # Vorwärtsdurchlauf
        loss   = criterion(output, y_batch)     # Verlust berechnen
        loss.backward()                         # Rückwärtsdurchlauf (Gradienten)
        optimizer.step()                        # Gewichte aktualisieren
    scheduler.step()                            # Lernrate anpassen

    # Validierung nach jeder Epoche
    nn_model.eval()
    with torch.no_grad():
        val_out  = nn_model(X_test_t)
        val_pred = torch.argmax(val_out, dim=1).cpu().numpy()
        val_acc  = sk_acc(y_test, val_pred)

    # Bestes Modell speichern
    if val_acc > best_acc:
        best_acc   = val_acc
        no_improve = 0
        torch.save(nn_model.state_dict(), f"{MODEL_ORDNER}/best_nn_model.pt")
    else:
        no_improve += 1

    # Early Stopping prüfen
    if no_improve >= patience:
        print(f"Early Stopping bei Epoch {epoch+1} (beste Accuracy: {best_acc*100:.2f}%)")
        break

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>3}/{EPOCHS} | Val Accuracy: {val_acc*100:.2f}%")

# Bestes Modell laden für spätere Nutzung
nn_model.load_state_dict(torch.load(f"{MODEL_ORDNER}/best_nn_model.pt"))
print("Training abgeschlossen")

# ──────────────────────────────────────────────────────────────────────────────
# 8. MODELLE SPEICHERN
#    Alle Modelle werden gespeichert damit sie in evaluation.py
#    wiederverwendet werden können ohne erneut zu trainieren.
#    - sklearn-Modelle → .pkl (joblib)
#    - PyTorch-Modell  → .pt  (state_dict)
#    - Scaler + Encoder werden ebenfalls gespeichert, da sie beim
#      späteren Vorhersagen auf neue Daten angewendet werden müssen.
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_ORDNER, exist_ok=True)

joblib.dump(rf,     f"{MODEL_ORDNER}/random_forest.pkl")
joblib.dump(svm,    f"{MODEL_ORDNER}/svm.pkl")
joblib.dump(knn,    f"{MODEL_ORDNER}/knn.pkl")
joblib.dump(scaler, f"{MODEL_ORDNER}/scaler.pkl")
joblib.dump(le,     f"{MODEL_ORDNER}/label_encoder.pkl")

# Testdaten speichern → werden in evaluation.py verwendet
np.save(f"{MODEL_ORDNER}/X_test.npy", X_test)
np.save(f"{MODEL_ORDNER}/y_test.npy", y_test)

print("\n" + "=" * 50)
print("  Training abgeschlossen!")
print(f"  Alle Modelle gespeichert in: ./{MODEL_ORDNER}/")
print("=" * 50)
print("\nNächster Schritt: python evaluation.py")