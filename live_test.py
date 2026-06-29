"""
================================================
 Live Handgestenerkennung — Echtzeit-Test
================================================
Starten:  python live_test.py
Beenden:  Taste 'q'

Modell wechseln:
  r = Random Forest
  s = SVM
  k = KNN
  n = Neuronales Netz
================================================
"""

import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH     = "yolo/best.pt"
MODEL_ORDNER   = "modelle"
CONF_THRESHOLD = 0.5

# Direkte Zuordnung: Klassennummer → Name (kein LabelEncoder nötig)
KLASSEN_NAMEN = {
    0: "hand_offen",
    1: "faust",
    2: "daumen_hoch",
    3: "zeigefinger",
    4: "winken",
    5: "kreisbewegung"
}

FARBEN = {
    "hand_offen":    (0, 255, 0),
    "faust":         (0, 0, 255),
    "daumen_hoch":   (255, 165, 0),
    "zeigefinger":   (255, 0, 255),
    "winken":        (0, 255, 255),
    "kreisbewegung": (255, 255, 0),
}

MODELL_NAMEN = {
    "rf":  "Random Forest",
    "svm": "SVM",
    "knn": "KNN",
    "nn":  "Neuronales Netz"
}

# ──────────────────────────────────────────────────────────────────────────────
# NEURONALES NETZ
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

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE BERECHNUNG — identisch mit aufnahme.py
# ──────────────────────────────────────────────────────────────────────────────

def berechne_distanz(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def berechne_winkel(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
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

# ──────────────────────────────────────────────────────────────────────────────
# MODELLE LADEN
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 50)
print("  Live Handgestenerkennung")
print("=" * 50)

print("\nLade ML-Modelle...")
rf     = joblib.load(f"{MODEL_ORDNER}/random_forest.pkl")
svm    = joblib.load(f"{MODEL_ORDNER}/svm.pkl")
knn    = joblib.load(f"{MODEL_ORDNER}/knn.pkl")
scaler = joblib.load(f"{MODEL_ORDNER}/scaler.pkl")

nn_model = HandGestenNN(10, len(KLASSEN_NAMEN))
nn_model.load_state_dict(torch.load(f"{MODEL_ORDNER}/best_nn_model.pt",
                                     map_location="cpu"))
nn_model.eval()
print("✓ Random Forest, SVM, KNN, Neuronales Netz geladen")

print("Lade YOLO Hand-Modell...")
yolo = YOLO(MODEL_PATH)
print("✓ YOLO geladen")

# ──────────────────────────────────────────────────────────────────────────────
# VORHERSAGE — gibt direkt Klassenname zurück
# ──────────────────────────────────────────────────────────────────────────────

def vorhersage(features, modell="rf"):
    X = scaler.transform([features])

    if modell == "rf":
        pred = int(rf.predict(X)[0])
        prob = rf.predict_proba(X)[0].max()
    elif modell == "svm":
        pred = int(svm.predict(X)[0])
        prob = svm.predict_proba(X)[0].max()
    elif modell == "knn":
        pred = int(knn.predict(X)[0])
        prob = knn.predict_proba(X)[0].max()
    elif modell == "nn":
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out   = nn_model(X_t)
            probs = torch.softmax(out, dim=1)
            pred  = int(torch.argmax(probs, dim=1).item())
            prob  = float(probs[0][pred].item())

    # Klassennummer direkt in Namen umwandeln
    name = KLASSEN_NAMEN.get(pred, f"klasse_{pred}")
    return name, prob

# ──────────────────────────────────────────────────────────────────────────────
# LIVE LOOP
# ──────────────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Fehler: Kamera nicht gefunden!")
    exit()

print("\n✓ Kamera geöffnet")
print("Tasten: [r] RF | [s] SVM | [k] KNN | [n] NN | [q] Beenden\n")

aktives_modell = "rf"
prev_kps       = None
geste          = "Keine Hand erkannt"
konfidenz      = 0.0
farbe          = (128, 128, 128)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    results = yolo.predict(source=frame, conf=CONF_THRESHOLD,
                           verbose=False, save=False)

    hand_erkannt = False

    if results and results[0].keypoints and results[0].keypoints.shape[1] > 0:
        kp_data      = results[0].keypoints
        keypoints_xy = kp_data.xy[0].cpu().numpy()

        if len(keypoints_xy) == 21:
            hand_erkannt = True

            for i, (x, y) in enumerate(keypoints_xy):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            try:
                features = berechne_features(keypoints_xy, prev_kps)
                geste, konfidenz = vorhersage(features, aktives_modell)
                farbe = FARBEN.get(geste, (255, 255, 255))
            except Exception as e:
                print(f"Fehler ({aktives_modell}): {e}")
                geste     = "Fehler"
                konfidenz = 0.0
                farbe     = (128, 128, 128)

            prev_kps = keypoints_xy.copy()
    else:
        geste     = "Keine Hand erkannt"
        konfidenz = 0.0
        farbe     = (128, 128, 128)
        prev_kps  = None

    # ── UI ────────────────────────────────────────────────────────────────────
    modell_name = MODELL_NAMEN[aktives_modell]
    hand_text   = "Hand erkannt" if hand_erkannt else "Keine Hand"
    hand_farbe  = (0, 220, 0) if hand_erkannt else (0, 0, 220)

    cv2.rectangle(frame, (0, 0), (w, 65), (0, 0, 0), -1)

    cv2.putText(frame, f"Geste: {geste}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, farbe, 2)

    cv2.putText(frame, f"Konfidenz: {konfidenz*100:.1f}%  |  Modell: {modell_name}",
                (10, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.putText(frame, hand_text,
                (8, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hand_farbe, 1)

    cv2.putText(frame, "[r] RF  [s] SVM  [k] KNN  [n] NN  [q] Beenden",
                (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    cv2.namedWindow("Live Handgestenerkennung", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("Live Handgestenerkennung", frame)
    cv2.resizeWindow("Live Handgestenerkennung", 1280, 720)

    taste = cv2.waitKey(1) & 0xFF
    if taste == ord('q'):
        break
    elif taste == ord('r'):
        aktives_modell = "rf"
        print("Modell: Random Forest")
    elif taste == ord('s'):
        aktives_modell = "svm"
        print("Modell: SVM")
    elif taste == ord('k'):
        aktives_modell = "knn"
        print("Modell: KNN")
    elif taste == ord('n'):
        aktives_modell = "nn"
        print("Modell: Neuronales Netz")

cap.release()
cv2.destroyAllWindows()
print("Live-Test beendet.")