"""
================================================================
 Live-Statistik  Handgestenerkennung
================================================================
Misst im Live-Betrieb, wie gut ein Modell eine Geste erkennt,
unter denselben Bedingungen wie die Trainingsdaten
(10 fps, 50 Frames statisch / 100 Frames dynamisch).

Pro Geste:
 1. Einzelframe-Genauigkeit  -> vergleichbar mit den Testdaten
 2. Mehrheitsergebnis        -> haeufigste Klasse ueber alle Frames

Steuerung:
  r / s / k / n     Modell waehlen (wie in live_test.py)
  Pfeil rechts / w  naechste Geste
  Pfeil links  / e  vorherige Geste
  LEERTASTE         Messung starten (stoppt automatisch)
  q                 Beenden

Ergebnis wird an  statistik/live_statistik.csv  angehaengt.
================================================================
"""

import cv2
import time
import csv
import os
from collections import Counter

import numpy as np
import joblib
import torch
import torch.nn as nn
from ultralytics import YOLO

# ----------------------------------------------------------------
MODEL_PATH     = "yolo/best.pt"
MODEL_ORDNER   = "modelle"
CONF_THRESHOLD = 0.5
ZIEL_FPS       = 10
CSV_OUT        = "statistik/live_statistik.csv"

KLASSEN_NAMEN = {
    0: "hand_offen", 1: "faust", 2: "daumen_hoch",
    3: "zeigefinger", 4: "winken", 5: "kreisbewegung",
}

FRAMES_PRO_GESTE = {
    "hand_offen": 50, "faust": 50, "daumen_hoch": 50,
    "zeigefinger": 50, "winken": 100, "kreisbewegung": 100,
}

TESTDATEN_ACC = {"rf": 94.32, "svm": 88.56, "knn": 92.70, "nn": 88.73}
MODELL_NAMEN  = {"rf": "Random Forest", "svm": "SVM",
                 "knn": "KNN", "nn": "Neuronales Netz"}

# ----------------------------------------------------------------
class HandGestenNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.netz = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.netz(x)

# ----------------------------------------------------------------
def berechne_distanz(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def berechne_winkel(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def berechne_features(kp, prev=None):
    handgroesse = berechne_distanz(kp[0], kp[12]) + 1e-6
    d_dz = berechne_distanz(kp[4], kp[8])  / handgroesse
    d_dm = berechne_distanz(kp[4], kp[12]) / handgroesse
    d_dk = berechne_distanz(kp[4], kp[20]) / handgroesse
    w_z  = berechne_winkel(kp[5],  kp[6],  kp[8])
    w_m  = berechne_winkel(kp[9],  kp[10], kp[12])
    w_r  = berechne_winkel(kp[13], kp[14], kp[16])
    w_k  = berechne_winkel(kp[17], kp[18], kp[20])
    v_d  = berechne_distanz(kp[4], prev[4]) / handgroesse if prev is not None else 0.0
    v_z  = berechne_distanz(kp[8], prev[8]) / handgroesse if prev is not None else 0.0
    return [d_dz, d_dm, d_dk, w_z, w_m, w_r, w_k, v_d, v_z, handgroesse]

# ----------------------------------------------------------------
print("=" * 55)
print("  Live-Statistik  Handgestenerkennung")
print("=" * 55)

print("\nLade Modelle ...")
rf     = joblib.load(f"{MODEL_ORDNER}/random_forest.pkl")
svm    = joblib.load(f"{MODEL_ORDNER}/svm.pkl")
knn    = joblib.load(f"{MODEL_ORDNER}/knn.pkl")
scaler = joblib.load(f"{MODEL_ORDNER}/scaler.pkl")

nn_model = HandGestenNN(10, len(KLASSEN_NAMEN))
nn_model.load_state_dict(torch.load(f"{MODEL_ORDNER}/best_nn_model.pt",
                                    map_location="cpu"))
nn_model.eval()
print("  ML-Modelle geladen")

print("Lade YOLO ...")
yolo = YOLO(MODEL_PATH)
print("  YOLO geladen")

# ----------------------------------------------------------------
def vorhersage_index(features, modell):
    X = scaler.transform([features])
    if modell == "rf":
        return int(rf.predict(X)[0])
    if modell == "svm":
        return int(svm.predict(X)[0])
    if modell == "knn":
        return int(knn.predict(X)[0])
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return int(torch.argmax(nn_model(X_t), dim=1).item())

# ----------------------------------------------------------------
os.makedirs("statistik", exist_ok=True)
csv_neu = not os.path.exists(CSV_OUT)
csv_file = open(CSV_OUT, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)
if csv_neu:
    writer.writerow([
        "modell", "ziel_geste", "frames", "richtige_frames",
        "einzelframe_accuracy", "mehrheitsklasse", "mehrheit_richtig",
        "gemessene_fps", "dauer_sekunden", "testdaten_accuracy",
    ])

# ----------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, ZIEL_FPS)

if not cap.isOpened():
    print("Fehler: Kamera nicht gefunden!")
    raise SystemExit

print("\nSteuerung:")
print("  r / s / k / n   Modell (RF / SVM / KNN / NN)")
print("  Pfeil re / w    naechste Geste")
print("  Pfeil li / e    vorherige Geste")
print("  LEERTASTE       Messung starten")
print("  q               Beenden\n")

klassen_liste  = list(KLASSEN_NAMEN.values())
klassen_index  = 0
aktives_modell = "rf"

messung_laeuft   = False
gesammelte_preds = []
frames_ziel      = 0
prev_kps         = None
mess_start       = 0.0

def zielgeste():
    return klassen_liste[klassen_index]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    results = yolo.predict(source=frame, conf=CONF_THRESHOLD,
                           verbose=False, save=False)
    hand_erkannt = False

    if results and results[0].keypoints and results[0].keypoints.shape[1] > 0:
        kp_xy = results[0].keypoints.xy[0].cpu().numpy()
        if len(kp_xy) == 21:
            hand_erkannt = True
            for (x, y) in kp_xy:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            if messung_laeuft and len(gesammelte_preds) < frames_ziel:
                feats = berechne_features(kp_xy, prev_kps)
                gesammelte_preds.append(vorhersage_index(feats, aktives_modell))
            prev_kps = kp_xy.copy()
    else:
        prev_kps = None

    if messung_laeuft and len(gesammelte_preds) >= frames_ziel:
        dauer = time.time() - mess_start
        gemessene_fps = frames_ziel / dauer if dauer > 0 else 0.0
        ziel = zielgeste()
        ziel_idx = klassen_liste.index(ziel)
        richtige = sum(1 for p in gesammelte_preds if p == ziel_idx)
        einzel_acc = 100.0 * richtige / frames_ziel
        mehrheit_idx = Counter(gesammelte_preds).most_common(1)[0][0]
        mehrheit_name = KLASSEN_NAMEN[mehrheit_idx]
        mehrheit_richtig = (mehrheit_idx == ziel_idx)

        print("\n" + "-" * 50)
        print(f"Ergebnis | {MODELL_NAMEN[aktives_modell]} | Geste: {ziel}")
        print(f"  Frames:              {frames_ziel}")
        print(f"  Gemessene FPS:       {gemessene_fps:.1f}")
        print(f"  Dauer:               {dauer:.1f} s")
        print(f"  Einzelframe richtig: {richtige}/{frames_ziel} = {einzel_acc:.1f}%")
        print(f"  Testdaten-Referenz:  {TESTDATEN_ACC[aktives_modell]:.2f}%")
        print(f"  Mehrheitsklasse:     {mehrheit_name} "
              f"({'RICHTIG' if mehrheit_richtig else 'FALSCH'})")
        print("-" * 50 + "\n")

        writer.writerow([
            MODELL_NAMEN[aktives_modell], ziel, frames_ziel, richtige,
            round(einzel_acc, 1), mehrheit_name, int(mehrheit_richtig),
            round(gemessene_fps, 1), round(dauer, 1),
            TESTDATEN_ACC[aktives_modell],
        ])
        csv_file.flush()
        messung_laeuft = False
        gesammelte_preds = []

    # UI
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    ziel = zielgeste()
    if messung_laeuft:
        status = f"MESSUNG  {len(gesammelte_preds)}/{frames_ziel}"
        farbe = (0, 220, 0)
    else:
        status = "BEREIT  (LEERTASTE = Start)"
        farbe = (0, 200, 255)
    cv2.putText(frame, f"Modell: {MODELL_NAMEN[aktives_modell]}   Geste: {ziel}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2)
    hand_text = "Hand erkannt" if hand_erkannt else "Keine Hand"
    hand_farbe = (0, 220, 0) if hand_erkannt else (0, 0, 220)
    cv2.putText(frame, hand_text, (8, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_farbe, 1)
    cv2.putText(frame,
                "[r/s/k/n] Modell   [<- ->] Geste   [LEER] Start   [q] Ende",
                (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)

    cv2.namedWindow("Live-Statistik", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Live-Statistik", frame)
    cv2.resizeWindow("Live-Statistik", 1280, 720)

    taste = cv2.waitKeyEx(1)
    t = taste & 0xFF

    if t == ord('q'):
        break
    elif t == ord(' ') and not messung_laeuft:
        gesammelte_preds = []
        frames_ziel = FRAMES_PRO_GESTE[zielgeste()]
        prev_kps = None
        mess_start = time.time()
        messung_laeuft = True
        print(f"Messung gestartet: {zielgeste()} ({frames_ziel} Frames)")
    elif t == ord('r'):
        aktives_modell = "rf";  print("Modell: Random Forest")
    elif t == ord('s'):
        aktives_modell = "svm"; print("Modell: SVM")
    elif t == ord('k'):
        aktives_modell = "knn"; print("Modell: KNN")
    elif t == ord('n'):
        aktives_modell = "nn";  print("Modell: Neuronales Netz")
    elif t == ord('w'):
        klassen_index = (klassen_index + 1) % len(klassen_liste)
        print(f"Geste: {zielgeste()}")
    elif t == ord('e'):
        klassen_index = (klassen_index - 1) % len(klassen_liste)
        print(f"Geste: {zielgeste()}")
    # Pfeiltasten (Codes koennen je nach System variieren)
    elif taste in (65363, 2555904, 83):   # Pfeil rechts
        klassen_index = (klassen_index + 1) % len(klassen_liste)
        print(f"Geste: {zielgeste()}")
    elif taste in (65361, 2424832, 81):   # Pfeil links
        klassen_index = (klassen_index - 1) % len(klassen_liste)
        print(f"Geste: {zielgeste()}")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print("Beendet. Ergebnisse in:", CSV_OUT)
