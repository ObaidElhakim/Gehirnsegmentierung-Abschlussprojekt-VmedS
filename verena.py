import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

# -------------------------------------------------------------------------
# Funktion: Brain-Maske für einen Slice erstellen (Relative Threshold)
# -------------------------------------------------------------------------
def get_brain_mask(slice_img, threshold=0.05):
    """
    Erstellt eine binäre Maske für das Gehirn aus einem 2D-Slice.
    
    Parameters:
        slice_img : 2D numpy array
            Einzelner Slice des MRI
        threshold : float
            Bruchteil des Maximalwerts, der als Gehirngewebe gilt
    
    Returns:
        brain_mask_clean : 2D boolean array
            Maske, True = Gehirn, False = Hintergrund
    """
    # 1. Schwellenwert
    thresh_val = threshold * np.max(slice_img)
    mask = slice_img > thresh_val

    # 2. Löcher füllen
    mask_filled = binary_fill_holes(mask)

    # 3. Größte zusammenhängende Komponente behalten
    labels, num = label(mask_filled)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # Hintergrund ignorieren
    largest_label = sizes.argmax()
    brain_mask_clean = labels == largest_label

    return brain_mask_clean

# -------------------------------------------------------------------------
# Funktion: MRI Slice laden (korrekte Orientierung)
# -------------------------------------------------------------------------
def load_slice_corrected(path, name):
    """
    Lädt ein MRI, extrahiert den mittleren Slice und korrigiert die Orientierung.
    
    Returns:
        slice_img : 2D numpy array
    """
    print(f"\n--- Lade {name} ---")
    img = np.asarray(nib.load(path).dataobj)
    print(f"{name} Original Form:", img.shape)

    # Mittleren Slice auswählen (Z-Achse)
    z_idx = img.shape[2] // 2
    slice_img = img[:, :, z_idx].astype(float)

    # Transpose für korrekte X/Y-Darstellung in Matplotlib
    slice_img = np.transpose(slice_img)

    # Flip vertikal + horizontal, damit Gehirn korrekt oben ist
    slice_img = np.flipud(slice_img)
    slice_img = np.fliplr(slice_img)

    return slice_img

# -------------------------------------------------------------------------
# Parameter
# -------------------------------------------------------------------------
brain_threshold = 0.05          # relativer Schwellenwert
t1_upper_threshold = 1200       # Intensität oberhalb -> wahrscheinlich Schädel
flair_lower_threshold = 50      # Intensität unterhalb -> kein Gehirn

# -------------------------------------------------------------------------
# MRI Slices laden
# -------------------------------------------------------------------------
T1_slice = load_slice_corrected("data/pat13_reg_T1.nii", "T1")
FLAIR_slice = load_slice_corrected("data/pat13_reg_FLAIR.nii", "FLAIR")

# -------------------------------------------------------------------------
# Brain-Masken erstellen
# -------------------------------------------------------------------------
T1_mask = get_brain_mask(T1_slice, threshold=brain_threshold)
FLAIR_mask = get_brain_mask(FLAIR_slice, threshold=brain_threshold)

# Kombinierte Maske (OR)
brain_mask_combined = T1_mask | FLAIR_mask

# -------------------------------------------------------------------------
# Intensity-basiertes Skull-Stripping
# -------------------------------------------------------------------------
brain_mask_filtered = brain_mask_combined & (T1_slice < t1_upper_threshold)
brain_mask_filtered = brain_mask_filtered & (FLAIR_slice > flair_lower_threshold)
brain_mask_stripped = binary_fill_holes(brain_mask_filtered)

# -------------------------------------------------------------------------
# Maskierte Gehirnbilder erstellen
# -------------------------------------------------------------------------
T1_brain_stripped = T1_slice * brain_mask_stripped
FLAIR_brain_stripped = FLAIR_slice * brain_mask_stripped

# -------------------------------------------------------------------------
# Visualisierung: Original Slices + Skull-stripped Brain
# -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(20,5))

# T1 Original
axs[0].imshow(T1_slice, cmap="gray", origin="lower")
axs[0].set_title("T1 Original")
axs[0].axis("off")

# FLAIR Original
axs[1].imshow(FLAIR_slice, cmap="gray", origin="lower")
axs[1].set_title("FLAIR Original")
axs[1].axis("off")

# Brain-Maske Overlay
axs[2].imshow(T1_slice, cmap="gray", origin="lower")
axs[2].imshow(brain_mask_stripped, cmap="Reds", alpha=0.3)
axs[2].set_title("Skull-stripped Brain Mask")
axs[2].axis("off")

# Maskiertes Gehirn
axs[3].imshow(T1_brain_stripped, cmap="gray", origin="lower")
axs[3].set_title("T1 Skull-stripped Brain")
axs[3].axis("off")

plt.show()

print("\nT1 und FLAIR erfolgreich verarbeitet. Skull-stripped Brain erstellt.")

# -------------------------------------------------------------------------
# Erweiterung: Multi-Modal-Segmentierung mit T1 + FLAIR + R1 (3 Cluster)
# -------------------------------------------------------------------------

# Optional: drittes Bild (R1/IR) vom gleichen Patienten und gleichen Slice
# Passe den Dateinamen an, z.B. "pat13_reg_IR.nii" oder "pat13_reg_R1.nii"
R1_slice = load_slice_corrected("data/pat13_reg_IR.nii", "IR")

# Nur die Gehirn-Pixel betrachten (brain_mask_stripped)
brain_idx = np.where(brain_mask_stripped)
if brain_idx[0].size == 0:
    raise RuntimeError("Brain-Maske ist leer – die Segmentierung kann nicht durchgeführt werden.")

# Feature-Vektor: [T1, FLAIR, R1] für jeden Gehirn-Pixel
feat_T1   = T1_slice[brain_idx]
feat_FLAIR = FLAIR_slice[brain_idx]
feat_IR   = R1_slice[brain_idx]

# Zu Matrix (N, 3) stapeln
features = np.stack([feat_T1, feat_FLAIR, feat_IR], axis=1).astype(float)

# Normierung pro Kanal (damit alle Kontraste ungefähr gleiche Skala haben)
eps = 1e-6
features_norm = features.copy()
for k in range(3):
    mu = np.mean(features_norm[:, k])
    sigma = np.std(features_norm[:, k]) + eps
    features_norm[:, k] = (features_norm[:, k] - mu) / sigma

# -------------------------------------------------------------------------
# Einfaches K-Means (K=3) mit NumPy – keine zusätzlichen Libraries nötig
# -------------------------------------------------------------------------
K = 3
max_iter = 20

# Zentren initialisieren: wähle 3 zufällige Gehirn-Pixel als Start
rng = np.random.default_rng(0)  # fixierter Seed für reproduzierbare Ergebnisse
rand_idx = rng.choice(features_norm.shape[0], size=K, replace=False)
centers = features_norm[rand_idx, :]

for it in range(max_iter):
    # 1) jedem Punkt das nächste Zentrum zuordnen
    #    -> euklidische Distanz zu allen Zentren
    dists = np.linalg.norm(features_norm[:, None, :] - centers[None, :, :], axis=2)  # (N, K)
    labels = np.argmin(dists, axis=1)  # (N,)

    # 2) neue Zentren als Mittelwerte innerhalb jeder Klasse
    new_centers = np.zeros_like(centers)
    for k in range(K):
        mask_k = labels == k
        if np.any(mask_k):
            new_centers[k, :] = np.mean(features_norm[mask_k], axis=0)
        else:
            # falls ein Cluster leer wird -> zufälligen Punkt zuweisen
            new_centers[k, :] = features_norm[rng.integers(0, features_norm.shape[0])]

    # 3) Abbruch, wenn sich die Zentren nicht mehr stark ändern
    shift = np.linalg.norm(new_centers - centers)
    centers = new_centers
    if shift < 1e-3:
        break

print(f"K-Means nach {it+1} Iterationen konvergiert.")

# -------------------------------------------------------------------------
# Cluster zu CSF / GM / WM zuordnen anhand der mittleren T1-Intensität
# -------------------------------------------------------------------------
cluster_T1_means = []
for k in range(K):
    mean_T1_k = np.mean(feat_T1[labels == k]) if np.any(labels == k) else np.inf
    cluster_T1_means.append((k, mean_T1_k))

# Sortieren: niedrigste T1 -> CSF, mittlere -> GM, höchste -> WM
cluster_T1_means.sort(key=lambda x: x[1])
csf_label = cluster_T1_means[0][0]
gm_label  = cluster_T1_means[1][0]
wm_label  = cluster_T1_means[2][0]

print("Cluster-Zuordnung (nach T1-Mittelwert):")
print(f"  CSF (blau)  = Cluster {csf_label}")
print(f"  GM  (grün)  = Cluster {gm_label}")
print(f"  WM  (rot)   = Cluster {wm_label}")

# -------------------------------------------------------------------------
# Masken im Bildraum aufbauen
# -------------------------------------------------------------------------
csf_mask = np.zeros_like(T1_slice, dtype=bool)
gm_mask  = np.zeros_like(T1_slice, dtype=bool)
wm_mask  = np.zeros_like(T1_slice, dtype=bool)

csf_mask[brain_idx] = (labels == csf_label)
gm_mask[brain_idx]  = (labels == gm_label)
wm_mask[brain_idx]  = (labels == wm_label)

# sicherstellen, dass nichts außerhalb der Gehirnmaske liegt
csf_mask &= brain_mask_stripped
gm_mask  &= brain_mask_stripped
wm_mask  &= brain_mask_stripped


# -------------------------------------------------------------------------
# RGB-Segmentbild erzeugen: CSF=blau, GM=grün, WM=rot
# -------------------------------------------------------------------------
seg_rgb = np.zeros(T1_slice.shape + (3,), dtype=float)

# CSF -> blau
seg_rgb[csf_mask, 2] = 1.0
# GM  -> grün
seg_rgb[gm_mask, 1] = 1.0
# WM  -> rot
seg_rgb[wm_mask, 0] = 1.0



# -------------------------------------------------------------------------
# Darstellung: T1, FLAIR, R1 und Segmentierung
# -------------------------------------------------------------------------
fig3, axs3 = plt.subplots(1, 4, figsize=(22, 5))

# T1
axs3[0].imshow(T1_slice, cmap="gray", origin="lower")
axs3[0].set_title("T1")
axs3[0].axis("off")

# FLAIR
axs3[1].imshow(FLAIR_slice, cmap="gray", origin="lower")
axs3[1].set_title("FLAIR")
axs3[1].axis("off")

# R1
axs3[2].imshow(R1_slice, cmap="gray", origin="lower")
axs3[2].set_title("R1 / IR")
axs3[2].axis("off")

# Overlay: T1 + farbige Segmentierung
axs3[3].imshow(T1_slice, cmap="gray", origin="lower")
axs3[3].imshow(seg_rgb, alpha=0.6, origin="lower")
axs3[3].set_title("Segmentierung: CSF(blau), GM(grün), WM(rot)")
axs3[3].axis("off")

plt.tight_layout()
plt.show()

print("Multi-Modal-Segmentierung (T1+FLAIR+R1) in CSF/GM/WM abgeschlossen.")
