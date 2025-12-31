import numpy as np
from scipy.ndimage import binary_fill_holes, label

# Modul: Brain Masking & Skull-Stripping
# Isoliert das Intrakranium von extrakraniellen Strukturen.

def get_brain_mask(slice_uint8, thr_rel=0.10):
    """
    Erstellt eine binäre Maske basierend auf adaptivem Thresholding.
    1. Thresholding: Filtert Hintergrundrauschen (>10% Signalstärke).
    2. Hole-Filling: Integriert hypointense Areale (z.B. Ventrikel) in die Maske.
    3. Largest Component: Eliminiert isolierte Artefakte (Augen, Rauschen) durch
       Selektion des größten zusammenhängenden Objekts.
    """
    thresh = thr_rel * float(np.max(slice_uint8))
    m = slice_uint8 > thresh
    m = binary_fill_holes(m)

    lbl, num = label(m)
    if num == 0:
        return np.zeros_like(slice_uint8, dtype=bool)

    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0 # Label 0 ist Hintergrund
    return lbl == sizes.argmax()

def apply_center_constraint(mask, keep_frac=0.85):
    """
    Wendet eine zentrale Region of Interest (ROI) an.
    Nutzt die Zentrierung registrierter Bilder, um Randartefakte (Halsansatz, Meningen)
    deterministisch zu entfernen. keep_frac definiert den behaltenen Anteil.
    """
    h, w = mask.shape
    ch, cw = h // 2, w // 2
    hh, ww = int(h * keep_frac / 2), int(w * keep_frac / 2)

    m2 = np.zeros_like(mask, dtype=bool)
    m2[ch - hh: ch + hh, cw - ww: cw + ww] = True
    return mask & m2