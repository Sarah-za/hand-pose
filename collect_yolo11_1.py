"""
Masterarbeit — Datenerhebung mit YOLO11 Hand-Pose
===================================================
- YOLO11 mit 21 Hand-Keypoints (dein best.pt / last.pt)
- Alle 21 Sessions
- Kein festes Limit — du drückst Q wenn fertig
- Jede Session = eigene CSV Datei

Steuerung:
    LEERTASTE  = Aufnahme starten (30 Frames, stoppt automatisch)
    N          = Naechste Klasse
    B          = Vorherige Klasse
    Q          = Beenden & speichern
"""

import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════
#  KONFIGURATION — hier anpassen
# ══════════════════════════════════════════════════════════════════

MODEL_PATH = r"C:\Users\Latitude 5510\Desktop\Masterarbeit\yolo\best.pt"         # <- dein Modell hier eintragen (best.pt oder last.pt)
FRAMES_PRO_AUFNAHME = 30         # 30 Frames = ~1 Sekunde pro Aufnahme
ZIEL_AUFNAHMEN = 30              # Ziel pro Klasse pro Session
SPEICHERORT = "handtracking_thesis/data/raw"
CONF_SCHWELLE = 0.3              # Keypoints unter diesem Wert werden ignoriert

KLASSEN = {
    0: "faust",
    1: "hand_offen",
    2: "daumen_hoch",
    3: "zeigefinger",
    4: "kreisbewegung",
    5: "winken",
}

SESSIONS = {
    "1":  {"name": "Basis",           "bedingung": "Normales Licht, frontal, Abstand 40-60cm"},
    "2":  {"name": "Helles_Licht",    "bedingung": "Starke Lampe direkt auf Hand richten"},
    "3":  {"name": "Schwaches_Licht", "bedingung": "Lampe gedimmt, Abends oder Vorhang zu"},
    "4":  {"name": "Seitl_Licht",     "bedingung": "Lampe nur von links oder rechts"},
    "5":  {"name": "Gegenlicht",      "bedingung": "Fenster hinter der Hand"},
    "6":  {"name": "Nah_20-30cm",     "bedingung": "Hand sehr nah an der Kamera"},
    "7":  {"name": "Weit_70-100cm",   "bedingung": "Hand weit von der Kamera entfernt"},
    "8":  {"name": "Schraeg_links",   "bedingung": "Kamera leicht nach links versetzt (45 Grad)"},
    "9":  {"name": "Schraeg_rechts",  "bedingung": "Kamera leicht nach rechts versetzt (45 Grad)"},
    "10": {"name": "Von_oben",        "bedingung": "Kamera schaut von oben auf die Hand"},
    "11": {"name": "Handflaeche",     "bedingung": "Innenseite der Hand zeigt zur Kamera"},
    "12": {"name": "Handruecken",     "bedingung": "Rueckseite der Hand zeigt zur Kamera"},
    "13": {"name": "Hand_seitlich",   "bedingung": "Daumenseite zeigt zur Kamera (90 Grad)"},
    "14": {"name": "Hand_schraeg",    "bedingung": "Natuerliche Haltung, leicht schraeg"},
    "15": {"name": "Hell_Hintergr",   "bedingung": "Weisse Wand oder weisses Papier hinter Hand"},
    "16": {"name": "Dunkel_Hintergr", "bedingung": "Schwarzes Tuch oder dunkle Wand"},
    "17": {"name": "Unrein_Hintergr", "bedingung": "Tisch mit Gegensaenden, Regal im Hintergrund"},
    "18": {"name": "Proband",         "bedingung": "Andere Person, nur Basis-Bedingung"},
    "19": {"name": "Linke_Hand",      "bedingung": "Aufnahmen mit der linken Hand"},
    "20": {"name": "Langsam",         "bedingung": "Geste sehr langsam (nur Kreis und Winken)"},
    "21": {"name": "Schnell",         "bedingung": "Geste schnell (nur Kreis und Winken)"},
}

# ══════════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════

def spalten_namen():
    """21 Keypoints x 3 Werte (x, y, confidence) = 63 Spalten"""
    namen = []
    for i in range(21):
        namen.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_conf"])
    return namen

SPALTEN = spalten_namen() + [
    "klasse", "klasse_id", "session_id", "session_name", "aufnahme_id", "frame", "zeitstempel"
]

def session_csv_pfad(session_id, session_name):
    return f"{SPEICHERORT}/session_{str(session_id).zfill(2)}_{session_name}.csv"

def lade_session_daten(session_id, session_name):
    pfad = session_csv_pfad(session_id, session_name)
    if os.path.exists(pfad):
        df = pd.read_csv(pfad)
        print(f"  Vorhandene Daten geladen: {len(df) // FRAMES_PRO_AUFNAHME} Aufnahmen")
        return df
    return pd.DataFrame(columns=SPALTEN)

def zaehle_aufnahmen(df, klasse_id):
    if len(df) == 0:
        return 0
    return len(df[df["klasse_id"] == klasse_id]) // FRAMES_PRO_AUFNAHME

def naechste_aufnahme_id(df):
    if len(df) == 0 or "aufnahme_id" not in df.columns:
        return 0
    return int(df["aufnahme_id"].max()) + 1

def extrahiere_keypoints(ergebnis):
    """
    Extrahiert 21 Hand-Keypoints aus YOLO11-Ergebnis.
    Gibt Liste mit 63 Werten zurueck: [x0,y0,conf0, x1,y1,conf1, ...]
    Gibt None zurueck wenn keine Hand erkannt.
    """
    if ergebnis is None:
        return None
    kp_daten = ergebnis.keypoints
    if kp_daten is None or len(kp_daten.data) == 0:
        return None

    # Erste erkannte Hand nehmen
    kp = kp_daten.data[0].cpu().numpy()  # Shape: [21, 3]

    if len(kp) < 21:
        return None

    return kp[:21].flatten().tolist()  # 63 Werte

def zeichne_hand(frame, ergebnis, aufzeichnen=False):
    if ergebnis is None or ergebnis.keypoints is None:
        return frame

    # ✅ WICHTIG: keine Hand erkannt
    if len(ergebnis.keypoints.xy) == 0:
        return frame

    keypoints = ergebnis.keypoints.xy[0]
    confidences = ergebnis.keypoints.conf[0]

    # Live vs Aufnahme
    if aufzeichnen:
        radius = 6
        farbe = (0, 0, 255)   # rot
    else:
        radius = 3
        farbe = (0, 255, 0)   # grün

    for (x, y), conf in zip(keypoints, confidences):
        if conf < CONF_SCHWELLE:
            continue

        cv2.circle(
            frame,
            (int(x), int(y)),
            radius,
            farbe,
            -1
        )

    return frame


def zeige_status(frame, klasse_name, aufn_count, aufzeichnen,
                 frame_buffer_len, session_name, hand_ok):
    h, w = frame.shape[:2]

    # Obere Leiste
    cv2.rectangle(frame, (0, 0), (w, 115), (20, 20, 20), -1)
    # Untere Leiste
    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)

    # Session
    cv2.putText(frame, f"Session: {session_name}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Klasse
    cv2.putText(frame, f"Geste: {klasse_name.upper()}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Aufnahmen-Zaehler
    farbe = (0, 220, 100)
    cv2.putText(frame, f"Aufnahmen: {aufn_count} / {ZIEL_AUFNAHMEN}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, farbe, 2)

    # Ziel-Fortschrittsbalken
    bar_w = int(min(aufn_count / ZIEL_AUFNAHMEN, 1.0) * (w - 20))
    cv2.rectangle(frame, (10, 100), (10 + bar_w, 112), farbe, -1)
    cv2.rectangle(frame, (10, 100), (w - 10, 112), (80, 80, 80), 1)

    # Ziel erreicht
    if aufn_count >= ZIEL_AUFNAHMEN:
        cv2.putText(frame, "ZIEL ERREICHT!  Weiter mit N",
                    (w // 2 - 160, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # REC Anzeige
    if aufzeichnen:
        cv2.circle(frame, (w - 35, 30), 18, (0, 0, 220), -1)
        cv2.putText(frame, "REC", (w - 72, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        rec_b = int((frame_buffer_len / FRAMES_PRO_AUFNAHME) * (w - 20))
        cv2.rectangle(frame, (10, 100), (10 + rec_b, 112), (0, 0, 200), -1)

    # Hand nicht erkannt
    if not hand_ok:
        cv2.putText(frame, "Keine Hand erkannt!",
                    (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 220), 2)

    # Tastenbelegung unten
    cv2.putText(frame,
                "LEERTASTE=Aufnahme   N=Naechste   B=Vorherige   Q=Beenden",
                (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1)

# ══════════════════════════════════════════════════════════════════
#  SESSION AUSWAHL IM TERMINAL
# ══════════════════════════════════════════════════════════════════

def session_auswaehlen():
    os.makedirs(SPEICHERORT, exist_ok=True)

    print("\n" + "=" * 65)
    print("  MASTERARBEIT — DATENERHEBUNG (YOLO11)")
    print("=" * 65)
    print(f"  {'Nr':<5} {'Session':<22} {'Bedingung':<30} {'Stand'}")
    print("-" * 65)

    for sid, info in SESSIONS.items():
        pfad = session_csv_pfad(int(sid), info["name"])
        stand = ""
        if os.path.exists(pfad):
            df_check = pd.read_csv(pfad)
            aufn = len(df_check) // FRAMES_PRO_AUFNAHME
            stand = f"<-- {aufn} Aufn. vorhanden"
        print(f"  [{sid:<3}] {info['name']:<22} {info['bedingung'][:30]:<30} {stand}")

    print("=" * 65)
    print("\n  Tipp: Fuer Probanden '18' waehlen, dann Namen eingeben.")
    print()

    while True:
        auswahl = input("  Session-Nummer eingeben: ").strip()

        if auswahl not in SESSIONS:
            print(f"  Ungueltig. Bitte Zahl zwischen 1 und {len(SESSIONS)} eingeben.")
            continue

        info = SESSIONS[auswahl].copy()

        # Proband: Namen abfragen
        if auswahl == "18":
            name = input("  Name des Probanden (z.B. Max): ").strip()
            if name:
                info["name"] = f"Proband_{name}"

        print(f"\n  Gewaehlt: [{auswahl}] {info['name']}")
        print(f"  Bedingung: {info['bedingung']}")
        print(f"  Ziel: {ZIEL_AUFNAHMEN} Aufnahmen pro Klasse")
        return auswahl, info

# ══════════════════════════════════════════════════════════════════
#  HAUPTPROGRAMM
# ══════════════════════════════════════════════════════════════════

def main():
    # YOLO11 laden
    print(f"\n  Lade Modell: {MODEL_PATH} ...")
    modell = YOLO(MODEL_PATH)
    print(f"  Modell geladen!")

    # Session auswaehlen
    session_id, session_info = session_auswaehlen()
    session_name = session_info["name"]

    # Daten laden (weitermachen falls unterbrochen)
    df   = lade_session_daten(session_id, session_name)
    pfad = session_csv_pfad(int(session_id), session_name)

    # Kamera starten
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  FEHLER: Kamera nicht gefunden!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"\n  Kamera bereit. Viel Erfolg!\n")

    aktuell_klasse_idx = 0
    aufzeichnen        = False
    frame_buffer       = []
    aufnahme_id        = naechste_aufnahme_id(df)
    hand_ok            = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        klasse_id   = aktuell_klasse_idx
        klasse_name = KLASSEN[klasse_id]
        aufn_count  = zaehle_aufnahmen(df, klasse_id)

        # YOLO11 Inferenz
        ergebnisse = modell(frame, verbose=False)
        ergebnis   = ergebnisse[0] if ergebnisse else None

        # Keypoints extrahieren
        kp_liste = extrahiere_keypoints(ergebnis)
        hand_ok  = kp_liste is not None

        # Hand zeichnen
        frame = zeichne_hand(frame, ergebnis)

        # Aufnahme laeuft
        if aufzeichnen:
            if hand_ok:
                sp = spalten_namen()
                zeile = {sp[i]: kp_liste[i] for i in range(len(sp))}
                zeile.update({
                    "klasse":       klasse_name,
                    "klasse_id":    klasse_id,
                    "session_id":   session_id,
                    "session_name": session_name,
                    "aufnahme_id":  aufnahme_id,
                    "frame":        len(frame_buffer),
                    "zeitstempel":  datetime.now().isoformat(),
                })
                frame_buffer.append(zeile)

                # 30 Frames voll -> speichern
                if len(frame_buffer) >= FRAMES_PRO_AUFNAHME:
                    df = pd.concat([df, pd.DataFrame(frame_buffer)], ignore_index=True)
                    df.to_csv(pfad, index=False)
                    aufnahme_id += 1
                    frame_buffer = []
                    aufzeichnen  = False
                    aufn_count   = zaehle_aufnahmen(df, klasse_id)
                    status = "ZIEL!" if aufn_count >= ZIEL_AUFNAHMEN else f"{aufn_count}/{ZIEL_AUFNAHMEN}"
                    print(f"  [{klasse_name}] Aufnahme gespeichert -> {status}")
            else:
                # Hand verschwunden -> Aufnahme abbrechen
                frame_buffer = []
                aufzeichnen  = False
                print(f"  [{klasse_name}] Abgebrochen - Hand nicht erkannt!")

        # Status anzeigen
        zeige_status(frame, klasse_name, aufn_count, aufzeichnen,
                     len(frame_buffer), session_name, hand_ok)

        cv2.imshow(f"Masterarbeit | {session_name}", frame)

        # Tastatur
        taste = cv2.waitKey(1) & 0xFF

        if taste == ord('q'):
            break

        elif taste == ord(' '):
            if not aufzeichnen:
                if hand_ok:
                    aufzeichnen  = True
                    frame_buffer = []
                    print(f"  [{klasse_name}] Aufnahme {aufn_count + 1} gestartet...")
                else:
                    print("  WARNUNG: Keine Hand erkannt!")

        elif taste == ord('n'):
            aktuell_klasse_idx = (aktuell_klasse_idx + 1) % len(KLASSEN)
            aufzeichnen  = False
            frame_buffer = []
            print(f"  Klasse: {KLASSEN[aktuell_klasse_idx]}")

        elif taste == ord('b'):
            aktuell_klasse_idx = (aktuell_klasse_idx - 1) % len(KLASSEN)
            aufzeichnen  = False
            frame_buffer = []
            print(f"  Klasse: {KLASSEN[aktuell_klasse_idx]}")

    # Aufraumen
    cap.release()
    cv2.destroyAllWindows()

    # Abschluss-Statistik
    print("\n" + "=" * 55)
    print(f"  SESSION {session_id} ABGESCHLOSSEN: {session_name}")
    print("=" * 55)
    if len(df) > 0:
        print(f"  Gespeichert: {pfad}")
        print(f"\n  Aufnahmen pro Klasse:")
        for kid, kname in KLASSEN.items():
            count  = zaehle_aufnahmen(df, kid)
            fertig = "OK" if count >= ZIEL_AUFNAHMEN else "  "
            balken = ("+" * count + "." * max(0, ZIEL_AUFNAHMEN - count))[:30]
            print(f"  {fertig}  {kname:<20} {count:>3}/{ZIEL_AUFNAHMEN}  [{balken}]")
    else:
        print("  Keine Daten aufgenommen.")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
