import numpy as np
import matplotlib.pyplot as plt

# Modul: Visualisierung & Reporting
# Erstellt grafische Auswertungen und RGB-Overlays.

def create_rgb(seg):
    """
    Konvertiert diskrete Label-Map in RGB-Darstellung.
    Farbschema: 1(CSF)=Blau, 2(GM)=Grün, 3(WM)=Rot.
    """
    rgb = np.zeros(seg.shape + (3,), dtype=np.float32)
    rgb[seg == 1, 2] = 1.0
    rgb[seg == 2, 1] = 1.0
    rgb[seg == 3, 0] = 1.0
    return rgb

def plot_results(results_list):
    """
    Generiert ein 2x5 Dashboard zum Vergleich der Patienten.
    Zeigt T1, Maske, Segmentierungs-Overlay, Label-Karte und statistische Metadaten.
    """
    fig, axs = plt.subplots(2, 5, figsize=(22, 10))
    
    for r, res in enumerate(results_list):
        # Spalte 1: T1 Original
        axs[r, 0].imshow(res["t1"], cmap="gray", origin="lower")
        axs[r, 0].set_title(f"Pat{res['patient_id']} T1 (0..255), z={res['z']}")
        axs[r, 0].axis("off")

        # Spalte 2: Masken-Kontrolle
        axs[r, 1].imshow(res["t1"], cmap="gray", origin="lower")
        axs[r, 1].imshow(res["mask"], alpha=0.35, origin="lower")
        axs[r, 1].set_title("Brain Mask")
        axs[r, 1].axis("off")

        # Spalte 3: Segmentierung (RGB)
        axs[r, 2].imshow(res["t1"], cmap="gray", origin="lower")
        axs[r, 2].imshow(res["seg_rgb"], alpha=0.55, origin="lower")
        axs[r, 2].set_title("Segmentierung")
        axs[r, 2].axis("off")

        # Spalte 4: Label Map
        axs[r, 3].imshow(res["seg"], cmap="tab10", origin="lower")
        axs[r, 3].set_title("Labels (1=CSF, 2=GM, 3=WM)")
        axs[r, 3].axis("off")

        # Spalte 5: Statistik/Metadaten
        axs[r, 4].axis("off")
        ic = res["init_centers_unweighted"]
        fc = res["final_centers_unweighted"]
        txt = (
            "Init-Zentren (T1,FL,IR)\n"
            f"CSF: {np.round(ic[0],1)}\nGM : {np.round(ic[1],1)}\nWM : {np.round(ic[2],1)}\n\n"
            "Final-Zentren\n"
            f"CSF: {np.round(fc[0],1)}\nGM : {np.round(fc[1],1)}\nWM : {np.round(fc[2],1)}"
        )
        axs[r, 4].text(0.0, 0.5, txt, fontsize=10, va="center")

    plt.tight_layout()
    plt.show()