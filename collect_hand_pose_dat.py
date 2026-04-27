
"""
Masterarbeit — Datenerhebung mit YOLOv11 Hand-Keypoints
=====================================================
- 21 Hand-Keypoints
- Kurze Aufnahmen (30 Frames)
- Eine Aufnahme = genau eine Session + eine Geste
- Session & Klasse während REC gesperrt
- REC nur möglich, wenn Hand erkannt ist

Steuerung:
LEERTASTE = Aufnahme starten
N          = Nächste Geste
B          = Vorherige Geste
Q          = Beenden
"""

import cv2
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO

# ──────────────────────────────────────────────────
# KONFIGURATION
# ──────────────────────────────────────────────────

MODEL_PATH = r"C:\Users\Latitude 5510\Desktop\Masterarbeit\yolo\best.pt"
FRAMES_PRO_AUFNAHME = 30
CONF_SCHWELLE = 0.3
SPEICHERORT = "handtracking_thesis/data/raw"

# ✅ GEÄNDERTE KLASSEN-REIHENFOLGE
KLASSEN = {
    0: "hand_offen",
    1: "faust",
    2: "daumen_hoch",
    3: "zeigefinger",
    4: "kreisbewegung",
    5: "winken",
}

SESSIONS = {
    "1":  {"name": "Basis", "bedingung": "Normales Licht, frontal, Abstand 40–60cm"},
    "2":  {"name": "Helles_Licht", "bedingung": "Starke Lampe direkt auf Hand"},
    "3":  {"name": "Schwaches_Licht", "bedingung": "Gedimmtes Licht / Abend"},
    "4":  {"name": "Seitl_Licht", "bedingung": "Licht von links oder rechts"},
    "5":  {"name": "Nah", "bedingung": "Abstand 20–30 cm"},
    "6":  {"name": "Weit", "bedingung": "Abstand 70–100 cm"},
    "7":  {"name": "Schraeg_links", "bedingung": "Hand ca. 45° links"},
    "8":  {"name": "Schraeg_rechts", "bedingung": "Hand ca. 45° rechts"},
    "9":  {"name": "Heller_HG", "bedingung": "Weißer Hintergrund"},
    "10": {"name": "Dunkler_HG", "bedingung": "Dunkler Hintergrund"},
    "11": {"name": "Proband", "bedingung": "Andere Person"},
    "12": {"name": "Linke_Hand", "bedingung": "Linke Hand"},
    "13": {"name": "Langsam", "bedingung": "Langsam (nur Kreis & Winken)"},
    "14": {"name": "Normal", "bedingung": "Normales Tempo"},
    "15": {"name": "Schnell", "bedingung": "Schnell (nur Kreis & Winken)"},
}

# ──────────────────────────────────────────────────
# HILFSFUNKTIONEN
# ──────────────────────────────────────────────────

def spalten_namen():
    cols = []
    for i in range(21):
        cols.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"])
    return cols

SPALTEN = spalten_namen() + [
    "klasse", "klasse_id", "session_id", "session_name",
    "aufnahme_id", "frame", "zeitstempel"
]

def session_csv(session_id, name):
    return f"{SPEICHERORT}/session_{int(session_id):02d}_{name}.csv"

def extrahiere_keypoints(result):
    if result is None or result.keypoints is None:
        return None
    if len(result.keypoints.data) == 0:
        return None
    kp = result.keypoints.data[0].cpu().numpy()
    if kp.shape[0] < 21:
        return None
    return kp.flatten().tolist()

def zeichne_hand(frame, result, recording):
    if result is None or result.keypoints is None:
        return frame
    if len(result.keypoints.xy) == 0:
        return frame

    pts  = result.keypoints.xy[0]
    conf = result.keypoints.conf[0]

    radius = 6 if recording else 3
    color  = (0, 0, 255) if recording else (0, 255, 0)

    for (x, y), c in zip(pts, conf):
        if c >= CONF_SCHWELLE:
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)

    return frame

def session_auswahl():
    os.makedirs(SPEICHERORT, exist_ok=True)
    print("\nVerfügbare Sessions:")
    for k, v in SESSIONS.items():
        print(f"[{k}] {v['name']} – {v['bedingung']}")
    while True:
        s = input("Session wählen: ").strip()
        if s in SESSIONS:
            return s, SESSIONS[s]

# ──────────────────────────────────────────────────
# HAUPTPROGRAMM
# ──────────────────────────────────────────────────

def main():
    model = YOLO(MODEL_PATH)

    session_id, session_info = session_auswahl()
    session_name = session_info["name"]

    pfad = session_csv(session_id, session_name)
    df = pd.read_csv(pfad) if os.path.exists(pfad) else pd.DataFrame(columns=SPALTEN)
    aufnahme_id = 0 if len(df) == 0 else int(df["aufnahme_id"].max()) + 1

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    klasse_idx = 0
    recording = False
    buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        klasse_name = KLASSEN[klasse_idx]
        next_klasse = KLASSEN[(klasse_idx + 1) % len(KLASSEN)]

        result = model(frame, verbose=False)[0]
        kp = extrahiere_keypoints(result)
        hand_ok = kp is not None

        frame = zeichne_hand(frame, result, recording)

        # ── Aufnahme ───────────────────────────────
        if recording:
            if not hand_ok:
                buffer = []
                recording = False
                print("[ABBRUCH] Hand verloren")
            else:
                row = dict(zip(spalten_namen(), kp))
                row.update({
                    "klasse": klasse_name,
                    "klasse_id": klasse_idx,
                    "session_id": session_id,
                    "session_name": session_name,
                    "aufnahme_id": aufnahme_id,
                    "frame": len(buffer),
                    "zeitstempel": datetime.now().isoformat(),
                })
                buffer.append(row)

                if len(buffer) >= FRAMES_PRO_AUFNAHME:
                    df = pd.concat([df, pd.DataFrame(buffer)], ignore_index=True)
                    df.to_csv(pfad, index=False)
                    buffer = []
                    recording = False
                    aufnahme_id += 1
                    print(f"[OK] Aufnahme gespeichert – {klasse_name}")

        # ── Session-Anzeige ────────────────────────
        overlay = frame.copy()
        h, w = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame,
                    f"SESSION: {session_name.upper()}",
                    (w // 2 - 220, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2)

        # ── Status ─────────────────────────────────
        if recording:
            status_text = "REC"
            status_color = (0, 0, 255)
        elif hand_ok:
            status_text = "Hand erkannt"
            status_color = (0, 220, 0)
        else:
            status_text = "Keine Hand erkannt"
            status_color = (0, 140, 255)

        cv2.putText(frame, status_text, (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # ── Klassenanzeige ─────────────────────────
        cv2.putText(frame, f"GESTE: {klasse_name.upper()}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if not recording:
            cv2.putText(frame, f"NAECHSTE: {next_klasse}",
                        (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

        cv2.imshow("Hand Datenerhebung", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' ') and not recording:
            if hand_ok:
                recording = True
                buffer = []
                print(f"[REC] {klasse_name}")
            else:
                print("[WARNUNG] Keine Hand erkannt – REC gesperrt")

        elif key == ord('n') and not recording:
            klasse_idx = (klasse_idx + 1) % len(KLASSEN)

        elif key == ord('b') and not recording:
            klasse_idx = (klasse_idx - 1) % len(KLASSEN)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()