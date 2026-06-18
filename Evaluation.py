import numpy as np
import joblib
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# ── Konfiguration ──────────────────────────────────────────────────────────────
MODEL_ORDNER = "modelle"
ERGEBNIS_ORDNER = "ergebnisse"

KLASSEN_NAMEN = [
    "hand_offen", "faust", "daumen_hoch",
    "zeigefinger", "winken", "kreisbewegung"
]
# ──────────────────────────────────────────────────────────────────────────────

class HandGestenNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.netz = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.netz(x)

    def forward(self, x):
        return self.netz(x)


def lade_testdaten():
    X_test = np.load(f"{MODEL_ORDNER}/X_test.npy")
    y_test = np.load(f"{MODEL_ORDNER}/y_test.npy")
    return X_test, y_test


def evaluiere_modell(name, y_pred, y_test, inferenzzeit):
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score:      {f1:.4f}")
    print(f"  Inferenzzeit:  {inferenzzeit*1000:.2f} ms")
    print(f"\n{classification_report(y_test, y_pred, target_names=KLASSEN_NAMEN)}")
    return {"modell": name, "accuracy": acc, "f1": f1, "inferenzzeit_ms": inferenzzeit*1000}


def plot_confusion_matrix(name, y_test, y_pred):
    import os
    os.makedirs(ERGEBNIS_ORDNER, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=KLASSEN_NAMEN, yticklabels=KLASSEN_NAMEN)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Tatsächlich")
    plt.xlabel("Vorhergesagt")
    plt.tight_layout()
    dateiname = f"{ERGEBNIS_ORDNER}/confusion_{name.lower().replace(' ', '_')}.png"
    plt.savefig(dateiname)
    plt.close()
    print(f"  ✓ Gespeichert: {dateiname}")


def plot_vergleich(ergebnisse):
    import os
    os.makedirs(ERGEBNIS_ORDNER, exist_ok=True)

    namen    = [e["modell"] for e in ergebnisse]
    accuracy = [e["accuracy"] for e in ergebnisse]
    f1       = [e["f1"] for e in ergebnisse]
    zeiten   = [e["inferenzzeit_ms"] for e in ergebnisse]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(namen, accuracy, color=["#4C72B0","#DD8452","#55A868","#C44E52"])
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")

    axes[1].bar(namen, f1, color=["#4C72B0","#DD8452","#55A868","#C44E52"])
    axes[1].set_title("F1-Score (weighted)")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1-Score")

    axes[2].bar(namen, zeiten, color=["#4C72B0","#DD8452","#55A868","#C44E52"])
    axes[2].set_title("Inferenzzeit (ms)")
    axes[2].set_ylabel("Millisekunden")

    plt.tight_layout()
    plt.savefig(f"{ERGEBNIS_ORDNER}/modell_vergleich.png")
    plt.close()
    print(f"\n✓ Vergleichsdiagramm gespeichert: {ERGEBNIS_ORDNER}/modell_vergleich.png")


def main():
    print("=" * 50)
    print("Evaluation — Handgestenerkennung")
    print("=" * 50)

    X_test, y_test = lade_testdaten()
    input_dim   = X_test.shape[1]
    num_classes = len(KLASSEN_NAMEN)

    ergebnisse = []

    # ── Random Forest ──────────────────────────────────────────────────────────
    rf = joblib.load(f"{MODEL_ORDNER}/random_forest.pkl")
    start = time.time()
    y_pred_rf = rf.predict(X_test)
    zeit_rf = time.time() - start
    ergebnisse.append(evaluiere_modell("Random Forest", y_pred_rf, y_test, zeit_rf))
    plot_confusion_matrix("Random Forest", y_test, y_pred_rf)

    # ── SVM ───────────────────────────────────────────────────────────────────
    svm = joblib.load(f"{MODEL_ORDNER}/svm.pkl")
    start = time.time()
    y_pred_svm = svm.predict(X_test)
    zeit_svm = time.time() - start
    ergebnisse.append(evaluiere_modell("SVM", y_pred_svm, y_test, zeit_svm))
    plot_confusion_matrix("SVM", y_test, y_pred_svm)

    # ── KNN ───────────────────────────────────────────────────────────────────
    knn = joblib.load(f"{MODEL_ORDNER}/knn.pkl")
    start = time.time()
    y_pred_knn = knn.predict(X_test)
    zeit_knn = time.time() - start
    ergebnisse.append(evaluiere_modell("KNN", y_pred_knn, y_test, zeit_knn))
    plot_confusion_matrix("KNN", y_test, y_pred_knn)

    # ── Neuronales Netz ───────────────────────────────────────────────────────
    nn_model = HandGestenNN(input_dim, num_classes)
    nn_model.load_state_dict(torch.load(f"{MODEL_ORDNER}/best_nn_model.pt"))
    nn_model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    start = time.time()
    with torch.no_grad():
        outputs = nn_model(X_tensor)
        y_pred_nn = torch.argmax(outputs, dim=1).numpy()
    zeit_nn = time.time() - start
    ergebnisse.append(evaluiere_modell("Neuronales Netz", y_pred_nn, y_test, zeit_nn))
    plot_confusion_matrix("Neuronales Netz", y_test, y_pred_nn)

    # ── Vergleich ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("ZUSAMMENFASSUNG")
    print("=" * 50)
    print(f"{'Modell':<20} {'Accuracy':>10} {'F1-Score':>10} {'Zeit (ms)':>12}")
    print("-" * 55)
    for e in ergebnisse:
        print(f"{e['modell']:<20} {e['accuracy']:>10.4f} {e['f1']:>10.4f} {e['inferenzzeit_ms']:>12.2f}")

    plot_vergleich(ergebnisse)


if __name__ == "__main__":
    main()