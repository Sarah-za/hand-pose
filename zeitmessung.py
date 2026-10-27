"""
Zeitmessung: Wie lange dauert die Verarbeitung EINES Bildes im Live-Betrieb?
Gemessen wird pro Bild: Keypoint-Extraktion (YOLO) + Merkmale + Klassifikation.
Das Warten auf die Kamera zaehlt nicht mit.
Ablauf: Fuer jedes Modell automatisch 410 Bilder mit erkannter Hand
(die ersten 10 zum Aufwaermen werden nicht gewertet -> 400 Bilder).
Waehrenddessen die Gesten vor der Kamera ausfuehren.  q = Abbruch
"""
import cv2, time
import numpy as np, joblib, torch, torch.nn as nn
from ultralytics import YOLO

BILDER = 410   # 400 Bilder + 10 zum Aufwaermen

# Welche Geste bis zu welchem Zaehlerstand gezeigt wird (wie bei der Datenerhebung)
ABLAUF = [(60, "hand_offen"), (110, "faust"), (160, "daumen_hoch"),
          (210, "zeigefinger"), (310, "winken"), (410, "kreisbewegung")]

def aktuelle_geste(n):
    for grenze, geste in ABLAUF:
        if n < grenze:
            return geste
    return ABLAUF[-1][1]

class HandGestenNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.netz = nn.Sequential(
            nn.Linear(10, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 6))
    def forward(self, x):
        return self.netz(x)

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def winkel(a, b, c):
    v1, v2 = np.array(a) - np.array(b), np.array(c) - np.array(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def merkmale(kp, prev):   # identisch zu live_statistik.py
    s = dist(kp[0], kp[12]) + 1e-6
    return [dist(kp[4], kp[8]) / s, dist(kp[4], kp[12]) / s, dist(kp[4], kp[20]) / s,
            winkel(kp[5], kp[6], kp[8]), winkel(kp[9], kp[10], kp[12]),
            winkel(kp[13], kp[14], kp[16]), winkel(kp[17], kp[18], kp[20]),
            dist(kp[4], prev[4]) / s if prev is not None else 0.0,
            dist(kp[8], prev[8]) / s if prev is not None else 0.0, s]

# Modelle laden
scaler = joblib.load("modelle/scaler.pkl")
modelle = {"Random Forest": joblib.load("modelle/random_forest.pkl"),
           "SVM": joblib.load("modelle/svm.pkl"),
           "KNN": joblib.load("modelle/knn.pkl")}
nn_model = HandGestenNN()
nn_model.load_state_dict(torch.load("modelle/best_nn_model.pt", map_location="cpu"))
nn_model.eval()
yolo = YOLO("yolo/best.pt")

def klassifiziere(name, f):
    X = scaler.transform([f])
    if name == "Neuronales Netz":
        with torch.no_grad():
            return int(torch.argmax(nn_model(torch.tensor(X, dtype=torch.float32))))
    return int(modelle[name].predict(X)[0])

# Kamera wie in live_statistik.py
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 10)

ergebnis = []
for name in ["Random Forest", "SVM", "KNN", "Neuronales Netz"]:
    print(f"\nMessung: {name} – bitte Gesten ausfuehren ...")
    zeiten, prev = [], None
    input(f"Bereit fuer {name}? ENTER druecken ...")
    while len(zeiten) < BILDER:
        ok, frame = cap.read()                      # Warten auf Kamera (nicht gemessen)
        if not ok:
            raise SystemExit("Kamerafehler")
        t0 = time.perf_counter()                    # Stoppuhr START
        r = yolo.predict(source=frame, conf=0.5, verbose=False)
        kp = None
        if r and r[0].keypoints and r[0].keypoints.shape[1] > 0:
            kp = r[0].keypoints.xy[0].cpu().numpy()
        if kp is not None and len(kp) == 21:
            klassifiziere(name, merkmale(kp, prev))
            zeiten.append((time.perf_counter() - t0) * 1000)   # Stoppuhr STOPP
            prev = kp.copy()
        else:
            prev = None
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 95), (0, 0, 0), -1)
        cv2.putText(frame, f"{name}: {len(zeiten)}/{BILDER}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Geste: {aktuelle_geste(len(zeiten))}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        cv2.imshow("Zeitmessung", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit("Abgebrochen")
    z = np.array(zeiten[10:])                       # erste 10 Bilder = Aufwaermen
    zeile = [name, len(z), round(z.mean(), 1), round(z.max(), 1), int((z > 100).sum())]
    ergebnis.append(zeile)
    print(f"  Mittel {zeile[2]} ms | Max {zeile[3]} ms | ueber 100 ms: {zeile[4]} von {zeile[1]}")

cap.release()
cv2.destroyAllWindows()
print("\n===== Ergebnis (Verarbeitung pro Bild) =====")
for name, n, mittel, maximum, ueber in ergebnis:
    print(f"{name:<16} Mittel {mittel:6.1f} ms | Max {maximum:6.1f} ms | ueber 100 ms: {ueber} von {n}")