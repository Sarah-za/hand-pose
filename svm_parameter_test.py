"""
Kurzer Test: SVM mit anderen Werten fuer C und gamma.
Gleiche Daten, gleiche Aufteilung (Seed 42, stratifiziert) und
gleiche Normierung wie in training.py.

Ausfuehren im selben Ordner wie training.py:
    python svm_parameter_test.py
"""
import os
import glob
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

DATENSATZ_ORDNER = "datensatz"
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

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
    "handgroesse",
]

# Zu testende Werte
C_WERTE = [5, 10, 20]
GAMMA_WERTE = ["scale", 0.01, 1.0]

# ── Daten laden (wie training.py) ─────────────────────────────────────────────
dateien = sorted(glob.glob(os.path.join(DATENSATZ_ORDNER, "*.csv")))
df = pd.concat([pd.read_csv(d) for d in dateien], ignore_index=True)
X = df[FEATURE_COLS].values
y = LabelEncoder().fit_transform(df["klasse"].values)

X_temp, X_test_raw, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"gamma='scale' entspricht hier: {1.0 / (X_train.shape[1] * X_train.var()):.4f}\n")

# ── Test ──────────────────────────────────────────────────────────────────────
print(f"{'C':>4} {'gamma':>7} | {'Val-Acc':>8} {'Val-F1':>7} | "
      f"{'Test-Acc':>8} {'Test-F1':>7} | {'Training':>8}")
print("-" * 66)

for C in C_WERTE:
    for gamma in GAMMA_WERTE:
        svm = SVC(kernel="rbf", C=C, gamma=gamma,
                  class_weight="balanced", random_state=RANDOM_STATE)
        t0 = time.perf_counter()
        svm.fit(X_train, y_train)
        dauer = time.perf_counter() - t0

        p_val = svm.predict(X_val)
        p_test = svm.predict(X_test)
        print(f"{C:>4} {str(gamma):>7} | "
              f"{accuracy_score(y_val, p_val)*100:7.2f}% "
              f"{f1_score(y_val, p_val, average='weighted'):7.4f} | "
              f"{accuracy_score(y_test, p_test)*100:7.2f}% "
              f"{f1_score(y_test, p_test, average='weighted'):7.4f} | "
              f"{dauer:7.1f} s")