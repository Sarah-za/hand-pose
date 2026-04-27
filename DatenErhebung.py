import cv2
import csv
import os
import time
import numpy as np
from ultralytics import YOLO
from collections import deque

# ============================================================
# KONFIGURATION
# ============================================================
MODEL_PATH = r"C:\Users\Latitude 5510\Desktop\Masterarbeit\yolo\best.pt"       # trainiertes YOLO-Modell
CSV_PATH   = "datensatz.csv"      # wo die Daten gespeichert werden
AUFNAHMEN_PRO_KLASSE = 200        # wie viele Samples pro Bewegung
FRAMES_PRO_SAMPLE    = 10         # Frames für Zeitreihen (dynamische Gesten)
CONF_THRESHOLD       = 0.5        # YOLO Confidence-Schwelle

# ============================================================
# BEWEGUNGSKLASSEN — hier kannst du deine Klassen anpassen
# ============================================================
KLASSEN = {
    "0": "hand_offen",
    "1": "faust",
    "2": "daumen_hoch",
    "3": "zeigefinger",
    "4": "winken",
    "5": "kreisbewegung"
}

# ============================================================
# KEYPOINT-INDIZES (MediaPipe / YOLO Hand: 21 Punkte)
# ============================================================
# 0=Handgelenk, 4=Daumen, 8=Zeigefinger, 12=Mittelfinger
# 16=Ringfinger, 20=kleiner Finger
FINGERSPITZEN = [4, 8, 12, 16, 20]

# ============================================================
# CSV HEADER erstellen
# ============================================================
def erstelle_csv_header():
    header = ["klasse", "zeitstempel"]
    # Rohe Keypoints (21 Punkte x 3 = x, y, confidence)
    for i in range(21):
        header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"]
    # Features
    header += [
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
    return header

# ============================================================
# FEATURE BERECHNUNG
# ============================================================
def berechne_distanz(p1, p2):
    """Euklidische Distanz zwischen zwei Keypoints."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def berechne_winkel(p1, p2, p3):
    """Winkel an p2 zwischen den Punkten p1-p2-p3."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def berechne_features(keypoints, prev_keypoints=None):
    """Berechnet alle Features aus den 21 Keypoints."""
    kp = keypoints  # shape: (21, 2)

    # Distanzen
    d_daumen_zeigefinger = berechne_distanz(kp[4], kp[8])
    d_daumen_mittelfinger = berechne_distanz(kp[4], kp[12])
    d_daumen_kleinfinger = berechne_distanz(kp[4], kp[20])

    # Handgröße zur Normalisierung (Handgelenk zu Mittelfingerspitze)
    handgroesse = berechne_distanz(kp[0], kp[12]) + 1e-6

    # Normalisierte Distanzen
    d_daumen_zeigefinger /= handgroesse
    d_daumen_mittelfinger /= handgroesse
    d_daumen_kleinfinger /= handgroesse

    # Winkel der Finger (Knöchel → Mittelglied → Spitze)
    w_zeigefinger  = berechne_winkel(kp[5],  kp[6],  kp[8])
    w_mittelfinger = berechne_winkel(kp[9],  kp[10], kp[12])
    w_ringfinger   = berechne_winkel(kp[13], kp[14], kp[16])
    w_kleinfinger  = berechne_winkel(kp[17], kp[18], kp[20])

    # Geschwindigkeit (nur wenn vorheriger Frame vorhanden)
    if prev_keypoints is not None:
        v_daumen     = berechne_distanz(kp[4],  prev_keypoints[4])  / handgroesse
        v_zeigefinger = berechne_distanz(kp[8], prev_keypoints[8])  / handgroesse
    else:
        v_daumen      = 0.0
        v_zeigefinger = 0.0

    return [
        round(d_daumen_zeigefinger, 4),
        round(d_daumen_mittelfinger, 4),
        round(d_daumen_kleinfinger, 4),
        round(w_zeigefinger, 2),
        round(w_mittelfinger, 2),
        round(w_ringfinger, 2),
        round(w_kleinfinger, 2),
        round(v_daumen, 4),
        round(v_zeigefinger, 4),
        round(handgroesse, 2)
    ]

# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    # YOLO laden
    print("Lade YOLO-Modell...")
    model = YOLO(MODEL_PATH)

    # CSV vorbereiten
    datei_existiert = os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not datei_existiert:
        writer.writerow(erstelle_csv_header())
        print(f"Neue CSV erstellt: {CSV_PATH}")
    else:
        print(f"Daten werden zu bestehender CSV hinzugefügt: {CSV_PATH}")

    # Webcam starten
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_keypoints = None
    aktuelle_klasse = None
    aufnahme_laeuft = False
    aufnahmen_count = 0

    print("\n" + "="*50)
    print("DATENERHEBUNG — Handbewegungserkennung")
    print("="*50)
    print("\nDrücke eine Taste um Klasse zu wählen:")
    for key, name in KLASSEN.items():
        print(f"  [{key}] → {name}")
    print("\n  [SPACE] → Aufnahme starten/stoppen")
    print("  [q]     → Beenden")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO Inferenz
        results = model.predict(source=frame, conf=CONF_THRESHOLD,
                                verbose=False, save=False)

        hand_erkannt = False
        keypoints_xy = None

        if results and results[0].keypoints and \
           results[0].keypoints.shape[1] > 0:

            kp_data = results[0].keypoints
            keypoints_xy = kp_data.xy[0].cpu().numpy()      # (21, 2)
            keypoints_conf = kp_data.conf[0].cpu().numpy()  # (21,)

            if len(keypoints_xy) == 21:
                hand_erkannt = True

                # Keypoints zeichnen
                for i, (x, y) in enumerate(keypoints_xy):
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                # Aufnahme speichern
                if aufnahme_laeuft and aktuelle_klasse and \
                   aufnahmen_count < AUFNAHMEN_PRO_KLASSE:

                    features = berechne_features(keypoints_xy, prev_keypoints)

                    # Zeile: klasse + zeitstempel + rohe keypoints + features
                    zeile = [aktuelle_klasse, round(time.time(), 3)]
                    for i in range(21):
                        zeile += [
                            round(float(keypoints_xy[i][0]), 2),
                            round(float(keypoints_xy[i][1]), 2),
                            round(float(keypoints_conf[i]), 3)
                        ]
                    zeile += features
                    writer.writerow(zeile)
                    aufnahmen_count += 1

                prev_keypoints = keypoints_xy.copy()

        # ---- UI auf Frame zeichnen ----
        # Status-Box
        status_farbe = (0, 200, 0) if aufnahme_laeuft else (0, 0, 200)
        cv2.rectangle(frame, (0, 0), (640, 70), (30, 30, 30), -1)

        klasse_text = KLASSEN.get(aktuelle_klasse, "keine gewählt")
        cv2.putText(frame, f"Klasse: {klasse_text}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status_text = f"Aufnahme: {'LÄUFT' if aufnahme_laeuft else 'GESTOPPT'}"
        cv2.putText(frame, status_text,
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_farbe, 2)

        count_text = f"Samples: {aufnahmen_count}/{AUFNAHMEN_PRO_KLASSE}"
        cv2.putText(frame, count_text,
                    (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        hand_text = "Hand erkannt" if hand_erkannt else "Keine Hand"
        hand_farbe = (0, 255, 0) if hand_erkannt else (0, 0, 255)
        cv2.putText(frame, hand_text,
                    (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_farbe, 2)

        # Fortschrittsbalken
        if aufnahmen_count > 0:
            progress = int((aufnahmen_count / AUFNAHMEN_PRO_KLASSE) * 620)
            cv2.rectangle(frame, (10, 460), (630, 475), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 460), (10 + progress, 475), (0, 200, 0), -1)

        # Automatisch stoppen wenn genug Samples
        if aufnahmen_count >= AUFNAHMEN_PRO_KLASSE and aufnahme_laeuft:
            aufnahme_laeuft = False
            print(f"\n✅ {AUFNAHMEN_PRO_KLASSE} Samples für '{klasse_text}' gesammelt!")
            print("Wähle nächste Klasse oder drücke [q] zum Beenden.")

        cv2.imshow("Datenerhebung — Handbewegungen", frame)

        # Tasteneingabe
        taste = cv2.waitKey(1) & 0xFF

        if taste == ord('q'):
            break
        elif taste == ord(' '):
            if aktuelle_klasse is None:
                print("⚠️  Bitte zuerst eine Klasse wählen (0-5)!")
            else:
                aufnahme_laeuft = not aufnahme_laeuft
                if aufnahme_laeuft:
                    aufnahmen_count = 0
                    print(f"\n▶  Aufnahme gestartet für: {KLASSEN[aktuelle_klasse]}")
                else:
                    print(f"⏸  Aufnahme gestoppt. {aufnahmen_count} Samples gespeichert.")
        elif chr(taste) in KLASSEN:
            aktuelle_klasse = chr(taste)
            aufnahme_laeuft = False
            aufnahmen_count = 0
            print(f"\n🖐  Klasse gewählt: {KLASSEN[aktuelle_klasse]}")

    # Aufräumen
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n✅ Fertig! Daten gespeichert in: {CSV_PATH}")

if __name__ == "__main__":
    main()
