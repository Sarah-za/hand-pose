# hand-pose

**Kamerabasierte Handgestenerkennung mit klassischen Machine-Learning-Verfahren**

Dieses Repository enthält den Code zur Masterarbeit

> *Evaluation und Vergleich verschiedener Machine-Learning-Methoden zur kamerabasierten Handgestenerkennung mit Anwendungspotenzial in Robotik und Prothetik*
>
> Sarah Zalloukh · Technische Hochschule Mannheim

Das System erkennt sechs Handgesten aus den Bildern einer normalen Webcam. Aus jedem Bild werden mit YOLOv8-Pose 21 Handkeypoints bestimmt, daraus zehn Merkmale berechnet und anschließend mit vier Verfahren klassifiziert: Random Forest, Support Vector Machine, K-Nearest Neighbors und ein vollverbundenes neuronales Netz.

---

## Verarbeitungskette

```
Kamerabild ─► YOLOv8-Pose (21 Keypoints) ─► 10 Merkmale ─► Klassifikator ─► erkannte Geste
```

**Merkmale:** 3 normierte Distanzen (Daumenspitze zu Zeige-, Mittel- und Kleinfinger), 4 Beugewinkel am PIP-Gelenk, 2 Geschwindigkeiten (Daumen- und Zeigefingerspitze zum vorherigen Bild), Handgröße.

**Gesten:** `hand_offen`, `faust`, `daumen_hoch`, `zeigefinger` (statisch) sowie `winken`, `kreisbewegung` (dynamisch).

---

## Ergebnisse

| Modell | Accuracy (Test) | F1 (gewichtet) | Accuracy (live) | Verarbeitung pro Bild live (Ø / max.) |
|---|---|---|---|---|
| Random Forest | **94,32 %** | **0,9433** | **94,9 %** | 52,5 ms / 62,7 ms |
| Support Vector Machine | 88,56 % | 0,8863 | 89,2 % | 20,0 ms / 21,6 ms |
| K-Nearest Neighbors | 92,70 % | 0,9268 | 90,0 % | 23,0 ms / 28,2 ms |
| Neuronales Netz | 88,73 % | 0,8876 | 87,3 % | 21,0 ms / 23,1 ms |

- **Datensatz:** 11.978 Bilder, 3 Probanden, 10 Aufnahmebedingungen, 6 Gesten
- **Aufteilung:** 64 % Training, 16 % Validierung, 20 % Test (stratifiziert, Seed 42)
- **Live-Auswertung:** neue Person, 72 Messungen
- **Echtzeit:** Bei allen Modellen wurde jedes Bild in weniger als 100 ms verarbeitet (10 Bilder pro Sekunde).

---

## Projektstruktur

```
hand-pose/
├── aufnahme.py              Datenerhebung
├── Train_models.py          Training aller vier Modelle
├── Evaluation.py            Evaluation und Diagrammerstellung
├── live_test.py             Echtzeitanwendung
├── live_statistik.py        Live-Auswertung (Genauigkeit)
├── zeitmessung.py           Verarbeitungszeit pro Bild im Live-Betrieb
├── svm_parameter_test.py    Test anderer Werte für C und gamma
├── yolo/
│   └── best.pt              trainiertes YOLOv8-Pose-Modell
├── datensatz/               CSV-Dateien je Proband und Session
├── modelle/                 gespeicherte Modelle, Scaler und Testdaten
├── statistik/               Ergebnisse der Live-Auswertung
└── ergebnisse/              Konfusionsmatrizen und Diagramme
```

---
## Installation

Getestet mit Python 3.13.

```bash
pip install ultralytics scikit-learn torch numpy opencv-python pandas joblib matplotlib seaborn
```

Für die Echtzeitanwendung wird eine Webcam benötigt.

---

## Verwendung

```bash
python aufnahme.py           # 1. Daten aufnehmen
python Train_models.py       # 2. Modelle trainieren
python Evaluation.py         # 3. Modelle auf den Testdaten bewerten
python live_test.py          # 4. Gesten live erkennen
python live_statistik.py     # 5. Live-Genauigkeit messen
python zeitmessung.py        # 6. Verarbeitungszeit pro Bild messen
```

Im Live-Betrieb wird das Modell per Taste gewählt: `r` Random Forest, `s` SVM, `k` KNN, `n` neuronales Netz, `q` beenden.

---

## Hardware

Alle Ergebnisse wurden auf einem Rechner mit Intel Core i9-12900, 32 GB RAM und NVIDIA RTX A4500 (20 GB) erzielt, mit einer Logitech C925e (1280 × 720, 10 Bilder pro Sekunde). Keypoint-Extraktion und das Training des neuronalen Netzes liefen auf der GPU, die Klassifikation auf der CPU.

---

## Datenschutz

Im Datensatz sind nur Keypoint-Koordinaten und daraus berechnete Merkmale gespeichert, keine Bilder. Die Probanden sind nur über die Kennungen P1 bis P3 zugeordnet.

