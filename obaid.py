"""
====================================================================================================
README: VERSION 1 - OPTIMIERTE SINGLE-SLICE PIPELINE
====================================================================================================

BESCHREIBUNG:
Diese Version stellt eine refactoring-optimierte Implementierung der Gehirnsegmentierung dar.
Der ursprüngliche Code wurde modularisiert und hinsichtlich der Laufzeit effizienter gestaltet.

WESENTLICHE ÄNDERUNGEN:
- Strukturierung: Aufteilung in funktionale Module (IO, Maskierung, Clustering).
- Robustheit: Implementierung relativer Schwellenwerte zur Verarbeitung variabler Signalintensitäten (Fix für Patient 13).
- Performance: Ersatz iterativer Schleifen durch vektorisierte NumPy-Operationen.

METHODIK:
1. Extraktion des zentralen axialen Slices (2D-Ansatz).
2. Multimodale Fusion von T1, FLAIR und IR Sequenzen.
3. Feature-Space-Konstruktion mit anschließender Z-Score Normalisierung.
4. Unüberwachtes Lernen mittels K-Means Clustering (K=3).
5. Automatische Zuordnung der Cluster zu Gewebeklassen (CSF/GM/WM) basierend auf T1-Intensität.

LIMITIERUNG:
Die Beschränkung auf einen einzelnen Slice (2D) macht das Verfahren anfällig für lokales Bildrauschen.
Eine Erweiterung auf volumetrische Analysen erfolgt in Version 2.
====================================================================================================
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# -------------------------------------------------------------------------
# MODUL 1: Vorverarbeitung & Maskierung
# -------------------------------------------------------------------------
def get_brain_mask(slice_img, threshold=0.05):
    """
    Erstellt eine binäre Maske zur Extraktion des intrakraniellen Volumens (Skull-Stripping).
    """
    # 1. Adaptives Thresholding: Berechnung eines relativen Schwellenwerts (5% des Maximums).
    # Dies gewährleistet Unabhängigkeit von absoluten Signalstärken.
    thresh_val = threshold * np.max(slice_img)
    mask = slice_img > thresh_val
    
    # 2. Morphologische Operation: Füllen von Löchern (binary_fill_holes).
    # Notwendig, da Ventrikel (Liquor) im T1-Bild dunkel erscheinen und sonst maskiert würden.
    mask_filled = binary_fill_holes(mask)
    
    # 3. Zusammenhangsanalyse (Connected Components):
    # Identifikation und Extraktion des größten zusammenhängenden Objekts (Gehirn),
    # um isolierte Artefakte (z.B. Augen, Rauschen) zu entfernen.
    labels, num = label(mask_filled)
    if num == 0: return np.zeros_like(slice_img, dtype=bool)
    
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0 # Hintergrund (Label 0) ignorieren
    largest_label = sizes.argmax()
    
    return labels == largest_label

# -------------------------------------------------------------------------
# MODUL 2: Daten-IO & Orientierungskorrektur
# -------------------------------------------------------------------------
def load_slice_corrected(path, name):
    """
    Lädt NIfTI-Volumendaten und extrahiert den mittleren 2D-Schnitt.
    """
    print(f"--- Lade {name} ---")
    img = np.asarray(nib.load(path).dataobj)
    
    # Auswahl des mittleren Slices der Z-Achse. Dieser Schnitt bietet in der Regel
    # eine repräsentative Darstellung der relevanten Anatomie (Basalganglien/Ventrikel).
    z_idx = img.shape[2] // 2
    slice_img = img[:, :, z_idx].astype(float)
    
    # Transformation der Matrix-Indizes in die radiologische Standardansicht
    # (Links=Rechts, Superior=Oben) mittels Transposition und Spiegelung.
    slice_img = np.transpose(slice_img)
    slice_img = np.flipud(slice_img)
    slice_img = np.fliplr(slice_img)
    
    return slice_img

# -------------------------------------------------------------------------
# MODUL 3: Haupt-Pipeline (Clustering & Segmentierung)
# -------------------------------------------------------------------------
def process_patient_full_pipeline(t1_path, flair_path, ir_path, patient_id):
    """
    Zentraler Workflow für die Single-Slice Segmentierung eines Patienten.
    """
    print(f"\n=== START PIPELINE: PATIENT {patient_id} ===")
    
    # 1. Laden der multimodalen Daten
    T1_slice = load_slice_corrected(t1_path, f"T1_pat{patient_id}")
    FLAIR_slice = load_slice_corrected(flair_path, f"FLAIR_pat{patient_id}")
    R1_slice = load_slice_corrected(ir_path, f"IR_pat{patient_id}")

    # 2. Erstellung der Gehirnmaske
    # Kombination von T1 und FLAIR Masken (ODER-Logik) zur Vermeidung von Informationsverlust
    # bei pathologischen Veränderungen, die in einer Sequenz isointens sein könnten.
    T1_mask = get_brain_mask(T1_slice)
    FLAIR_mask = get_brain_mask(FLAIR_slice)
    brain_mask_combined = T1_mask | FLAIR_mask

    # Zusätzliche Intensitätsfilterung zur Bereinigung (z.B. Entfernung von hellem Knochengewebe)
    t1_upper_threshold = 1200
    flair_lower_threshold = 50
    brain_mask_filtered = brain_mask_combined & (T1_slice < t1_upper_threshold)
    brain_mask_filtered = brain_mask_filtered & (FLAIR_slice > flair_lower_threshold)
    brain_mask_stripped = binary_fill_holes(brain_mask_filtered)

    # 3. Feature-Extraktion
    # Beschränkung der Berechnung auf Voxel innerhalb der Maske (ROI).
    brain_idx = np.where(brain_mask_stripped)
    if brain_idx[0].size == 0:
        raise RuntimeError(f"Brain-Maske für Patient {patient_id} ist leer!")

    # Konstruktion des 3D-Feature-Vektors pro Pixel [T1, FLAIR, IR]
    feat_T1 = T1_slice[brain_idx]
    feat_FLAIR = FLAIR_slice[brain_idx]
    feat_IR = R1_slice[brain_idx]
    features = np.stack([feat_T1, feat_FLAIR, feat_IR], axis=1).astype(float)

    # 4. Z-Score Normalisierung
    # Standardisierung der Features (Mittelwert 0, Std-Abweichung 1), um eine Verzerrung
    # des K-Means durch unterschiedliche Wertebereiche (T1 vs FLAIR) zu verhindern.
    eps = 1e-6
    features_norm = features.copy()
    for k in range(3):
        mu = np.mean(features_norm[:, k])
        sigma = np.std(features_norm[:, k]) + eps
        features_norm[:, k] = (features_norm[:, k] - mu) / sigma

    # 5. K-Means Clustering Implementation
    # Ziel: Trennung in K=3 Klassen (CSF, GM, WM).
    K = 3 
    rng = np.random.default_rng(0)
    
    # Initialisierung mit zufälligen Startzentren
    rand_idx = rng.choice(features_norm.shape[0], size=K, replace=False)
    centers = features_norm[rand_idx, :]

    for it in range(20):
        # Berechnung der euklidischen Distanz im Feature-Raum
        dists = np.linalg.norm(features_norm[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1) # Zuordnung zum nächsten Zentrum
        
        # Update der Zentren (Mittelwertbildung pro Cluster)
        new_centers = np.array([features_norm[labels == k].mean(axis=0) if np.any(labels == k) 
                               else features_norm[rng.integers(0, features_norm.shape[0])] for k in range(K)])
        
        # Konvergenzprüfung
        if np.linalg.norm(new_centers - centers) < 1e-3: break
        centers = new_centers

    # 6. Automatische Label-Zuordnung
    # Da K-Means Labels stochastisch vergibt, erfolgt eine Sortierung nach T1-Intensität:
    # Niedrig = CSF, Mittel = GM, Hoch = WM.
    cluster_T1_means = sorted([(k, np.mean(feat_T1[labels == k])) for k in range(K)], key=lambda x: x[1])
    csf_label, gm_label, wm_label = cluster_T1_means[0][0], cluster_T1_means[1][0], cluster_T1_means[2][0]

    # 7. Rekonstruktion des RGB-Segmentierungsbildes
    seg_rgb = np.zeros(T1_slice.shape + (3,), dtype=float)
    seg_rgb[brain_idx[0][labels == csf_label], brain_idx[1][labels == csf_label], 2] = 1.0 # Blau (CSF)
    seg_rgb[brain_idx[0][labels == gm_label], brain_idx[1][labels == gm_label], 1] = 1.0   # Grün (GM)
    seg_rgb[brain_idx[0][labels == wm_label], brain_idx[1][labels == wm_label], 0] = 1.0   # Rot (WM)

    return {
        "id": patient_id, "t1": T1_slice, "flair": FLAIR_slice, 
        "r1": R1_slice, "seg": seg_rgb, "mask": brain_mask_stripped,
        "t1_stripped": T1_slice * brain_mask_stripped
    }

# -------------------------------------------------------------------------
# EXECUTION & VISUALISIERUNG
# -------------------------------------------------------------------------
# Durchführung der Pipeline für beide Datensätze
results_pat7 = process_patient_full_pipeline("data/pat7_reg_T1.nii", "data/pat7_reg_FLAIR.nii", "data/pat7_reg_IR.nii", 7)
results_pat13 = process_patient_full_pipeline("data/pat13_reg_T1.nii", "data/pat13_reg_FLAIR.nii", "data/pat13_reg_IR.nii", 13)

# Visualisierung als 2x5 Grid
fig, axs = plt.subplots(2, 5, figsize=(22, 10))
fig.suptitle("Multimodale Gehirnsegmentierung (Version 1: Single-Slice)", fontsize=20)

for row, res in enumerate([results_pat7, results_pat13]):
    # Darstellung der Eingangsdaten
    axs[row, 0].imshow(res["t1"], cmap="gray", origin="lower"); axs[row, 0].set_title(f"Pat{res['id']}: T1")
    axs[row, 1].imshow(res["flair"], cmap="gray", origin="lower"); axs[row, 1].set_title(f"Pat{res['id']}: FLAIR")
    axs[row, 2].imshow(res["r1"], cmap="gray", origin="lower"); axs[row, 2].set_title(f"Pat{res['id']}: IR")

    # Overlay der Brain Mask zur Qualitätskontrolle
    axs[row, 3].imshow(res["t1_stripped"], cmap="gray", origin="lower")
    axs[row, 3].imshow(res["mask"], cmap="Reds", alpha=0.3, origin="lower")
    axs[row, 3].set_title("Brain Mask Overlay")

    # Finales Segmentierungsergebnis
    axs[row, 4].imshow(res["t1"], cmap="gray", origin="lower")
    axs[row, 4].imshow(res["seg"], alpha=0.6, origin="lower")
    axs[row, 4].set_title("Segmentierung (CSF/GM/WM)")

    for ax in axs[row]: ax.axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()