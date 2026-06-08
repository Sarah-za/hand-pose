import cv2
import csv
import os
import time
import numpy as np
from ultralytics import YOLO

# ============================================================
# KONFIGURATION
# ============================================================
MODEL_PATH           = "yolo/best.pt"
#CSV_PATH             = "datensatz.csv"
AUFNAHMEN_PRO_KLASSE = 100
CONF_THRESHOLD       = 0.5

# ============================================================
# BEWEGUNGSKLASSEN
# ============================================================
KLASSEN = {
    "0": {"name": "hand_offen",    "frames": 50},
    "1": {"name": "faust",         "frames": 50},
    "2": {"name": "daumen_hoch",   "frames": 50},
    "3": {"name": "zeigefinger",   "frames": 50},
    "4": {"name": "winken",        "frames": 100},
    "5": {"name": "kreisbewegung", "frames": 100},
}

# ============================================================
# SESSIONS — 10 Variationen (alle Probanden machen alle Sessions)
# Format: (Name, licht, abstand, hand)
# ============================================================
SESSIONS = {
    "1":  ("Basis",                 "normales_licht",   "40 cm",  "rechts"),
    "2":  ("Licht: Hell",           "helles_licht",     "40 cm",  "rechts"),
    "3":  ("Licht: Schwach",        "schwaches_licht",  "40 cm",  "rechts"),
    "4":  ("Licht: Seitlich",       "seitlich",         "40 cm",  "rechts"),
    "5":  ("Abstand: Nah",          "normales_licht",   "25 cm",  "rechts"),
    "6":  ("Abstand: Weit",         "normales_licht",   "60 cm", "rechts"),
    "7":  ("Rotation: Handflaeche", "normales_licht",   "40 cm",  "rechts"),
    "8":  ("Hintergrund: Hell",     "normales_licht",   "40 cm",  "rechts"),
    "9":  ("Hintergrund: Dunkel",   "normales_licht",   "40 cm",  "rechts"),
    "10": ("Linke Hand",            "normales_licht",   "40cm",  "links"),
}

# ============================================================
# CSV HEADER
# ============================================================
def erstelle_csv_header():
    header = ["proband", "session_id", "session_name", "licht", "abstand", "hand",
              "klasse", "zeitstempel"]
    for i in range(21):
        header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"]
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
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def berechne_winkel(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_a = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_a,-1.0,1.0)))

def berechne_features(kp, prev=None):
    handgroesse = berechne_distanz(kp[0], kp[12]) + 1e-6
    d_dz  = berechne_distanz(kp[4], kp[8])  / handgroesse
    d_dm  = berechne_distanz(kp[4], kp[12]) / handgroesse
    d_dk  = berechne_distanz(kp[4], kp[20]) / handgroesse
    w_z   = berechne_winkel(kp[5],  kp[6],  kp[8])
    w_m   = berechne_winkel(kp[9],  kp[10], kp[12])
    w_r   = berechne_winkel(kp[13], kp[14], kp[16])
    w_k   = berechne_winkel(kp[17], kp[18], kp[20])
    v_d   = berechne_distanz(kp[4], prev[4]) / handgroesse if prev is not None else 0.0
    v_z   = berechne_distanz(kp[8], prev[8]) / handgroesse if prev is not None else 0.0
    return [round(d_dz,4), round(d_dm,4), round(d_dk,4),
            round(w_z,2),  round(w_m,2),  round(w_r,2), round(w_k,2),
            round(v_d,4),  round(v_z,4),  round(handgroesse,2)]

# ============================================================
# HAUPTPROGRAMM
# ============================================================
def main():
    print("\n" + "="*55)
    print("  DATENERHEBUNG — Handbewegungserkennung")
    print("="*55)

    # Proband eingeben
    proband = input("\nProband-Name (z.B. P1 / P2 / P3): ").strip()
    if not proband:
        proband = "unbekannt"

    # Sessions anzeigen — alle machen alle 10 Sessions
    print(f"\nVerfuegbare Sessions fuer '{proband}':")
    print(f"{'Nr.':<5} {'Name':<25} {'Licht':<18} {'Abstand':<12} {'Hand'}")
    print("-"*75)
    for sid, (name, licht, abstand, hand) in SESSIONS.items():
        print(f"[{sid:>2}]  {name:<25} {licht:<18} {abstand:<12} {hand}")

    print(f"\nHinweis: Alle Probanden machen alle 10 Sessions.")
    statisch  = sum(1 for k in KLASSEN.values() if k["frames"] == 50)  * 50
    dynamisch = sum(1 for k in KLASSEN.values() if k["frames"] == 100) * 100
    gesamt    = statisch + dynamisch
    print(f"         {statisch} (statisch) + {dynamisch} (dynamisch) = {gesamt} Aufnahmen pro Session.")

    session_id = input("\nSession-Nummer eingeben: ").strip()
    if session_id not in SESSIONS:
        print("Ungueltige Session!")
        return
    os.makedirs("datensatz", exist_ok=True)
    CSV_PATH = f"datensatz/daten_{proband}_Session{session_id}.csv"
    session_name, licht, abstand, hand = SESSIONS[session_id]
    print(f"\nGewaehlt: Session {session_id} — {session_name}")
    print(f"Licht: {licht} | Abstand: {abstand} | Hand: {hand}")
    input("\nDruecke ENTER wenn du bereit bist...")

    # YOLO laden
    print("\nLade YOLO-Modell...")
    model = YOLO(MODEL_PATH)

    # CSV vorbereiten
    EXPECTED_COLS  = len(erstelle_csv_header())
    datei_existiert = os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not datei_existiert:
        writer.writerow(erstelle_csv_header())
        print(f"Neue CSV erstellt: {CSV_PATH}")
    else:
        print(f"Daten werden hinzugefuegt: {CSV_PATH}")

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam FPS: {fps}")

    klassen_liste   = list(KLASSEN.keys())
    klassen_index   = 0
    aktuelle_klasse = klassen_liste[0]
    prev_keypoints  = None
    aufnahme_laeuft = False
    aufnahmen_count = 0
    uebersprungen   = 0  # Zaehler fuer kaputte Zeilen

    print("\nSTEUERUNG:")
    print("  LEERTASTE = Aufnahme starten/stoppen")
    print("  N         = Naechste Geste")
    print("  B         = Vorherige Geste")
    print("  Q         = Beenden\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=CONF_THRESHOLD,
                                verbose=False, save=False)
        hand_erkannt   = False
        keypoints_xy   = None
        keypoints_conf = None

        if results and results[0].keypoints and results[0].keypoints.shape[1] > 0:
            kp_data        = results[0].keypoints
            keypoints_xy   = kp_data.xy[0].cpu().numpy()
            keypoints_conf = kp_data.conf[0].cpu().numpy()

            if len(keypoints_xy) == 21:
                hand_erkannt = True
                for i, (x, y) in enumerate(keypoints_xy):
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                if aufnahme_laeuft and aufnahmen_count < KLASSEN[aktuelle_klasse]["frames"]:
                    features = berechne_features(keypoints_xy, prev_keypoints)
                    zeile = [proband, session_id, session_name, licht, abstand, hand,
                             aktuelle_klasse, round(time.time(), 3)]
                    for i in range(21):
                        zeile += [round(float(keypoints_xy[i][0]),2),
                                  round(float(keypoints_xy[i][1]),2),
                                  round(float(keypoints_conf[i]),3)]
                    zeile += features

                    # ✅ BUG FIX: Nur speichern wenn Spaltenanzahl korrekt!
                    if len(zeile) == EXPECTED_COLS:
                        writer.writerow(zeile)
                        aufnahmen_count += 1
                    else:
                        uebersprungen += 1
                        print(f"⚠️  Zeile uebersprungen: {len(zeile)} statt {EXPECTED_COLS} Spalten")

                prev_keypoints = keypoints_xy.copy()

        # ---- UI ----
        h, w = frame.shape[:2]
        klasse_text  = KLASSEN[aktuelle_klasse]["name"]
        status_farbe = (0, 220, 0) if aufnahme_laeuft else (0, 0, 220)
        status_text  = "REC" if aufnahme_laeuft else "STOP"

        cv2.putText(frame, f"{klasse_text}  [{status_text}]",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_farbe, 2)
        cv2.putText(frame, f"{proband} | S{session_id}: {session_name}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, f"{aufnahmen_count}/{KLASSEN[aktuelle_klasse]["frames"]}",
            (w-130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        if aufnahmen_count > 0:
            prog = int((aufnahmen_count / KLASSEN[aktuelle_klasse]["frames"]) * w)
            cv2.rectangle(frame, (0, 48), (w, 55), (50, 50, 50), -1)
            cv2.rectangle(frame, (0, 48), (prog, 55), (0, 200, 0), -1)

        hand_text  = "Hand erkannt" if hand_erkannt else "Keine Hand"
        hand_farbe = (0, 220, 0) if hand_erkannt else (0, 0, 220)
        cv2.putText(frame, hand_text,
                    (8, h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, hand_farbe, 1)
        cv2.putText(frame, "LEERTASTE=Start/Stop   N=Naechste   B=Vorherige   Q=Beenden",
                    (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        if aufnahmen_count >= KLASSEN[aktuelle_klasse]["frames"] and aufnahme_laeuft:
            aufnahme_laeuft = False
            print(f"Fertig! {KLASSEN[aktuelle_klasse]["frames"]} Samples fuer '{klasse_text}' gesammelt!")
            if uebersprungen > 0:
                print(f"         ({uebersprungen} kaputte Zeilen uebersprungen)")
            uebersprungen = 0

        cv2.namedWindow("Datenerhebung", cv2.WINDOW_GUI_NORMAL)  
        cv2.imshow("Datenerhebung", frame)
        cv2.resizeWindow("Datenerhebung", 1280, 720) 
        taste = cv2.waitKey(1) & 0xFF

        if taste == ord('q'):
            break
        elif taste == ord(' '):
            aufnahme_laeuft = not aufnahme_laeuft
            if aufnahme_laeuft:
                aufnahmen_count = 0
                uebersprungen   = 0
                print(f"Aufnahme gestartet: {klasse_text}")
            else:
                print(f"Gestoppt. {aufnahmen_count} Samples gespeichert.")
        elif taste == ord('n'):
            klassen_index   = (klassen_index + 1) % len(klassen_liste)
            aktuelle_klasse = klassen_liste[klassen_index]
            aufnahme_laeuft = False
            aufnahmen_count = 0
            print(f"Naechste Geste: {KLASSEN[aktuelle_klasse]}")
        elif taste == ord('b'):
            klassen_index   = (klassen_index - 1) % len(klassen_liste)
            aktuelle_klasse = klassen_liste[klassen_index]
            aufnahme_laeuft = False
            aufnahmen_count = 0
            print(f"Vorherige Geste: {KLASSEN[aktuelle_klasse]}")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nFertig! Daten gespeichert in: {CSV_PATH}")

if __name__ == "__main__":
    main()