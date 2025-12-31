import numpy as np

# Modul: Daten-Laden und Normalisierung
# Verarbeitet rohe NIfTI-Daten zu standardisierten 8-Bit-Arrays.

def robust_minmax_to_uint8(vol, p_low=1, p_high=99):
    """
    Skaliert 3D-Volumendaten robust auf den Bereich 0-255 (uint8).
    Verwendet Perzentile (1% und 99%) statt Min/Max, um Signal-Spikes zu ignorieren,
    die bei MRT-Rohdaten häufig auftreten und den Gewebekontrast verfälschen würden.
    """
    v = vol.astype(np.float32)
    lo = np.percentile(v, p_low)
    hi = np.percentile(v, p_high)
    
    # Schutz vor Division durch Null bei leeren Volumina
    if hi <= lo + 1e-6:
        return np.zeros_like(v, dtype=np.uint8)

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    # Diskretisierung auf Ganzzahlen für Speicher- und Recheneffizienz
    return (v * 255.0).astype(np.uint8)

def orient_slice(vol_uint8, z):
    """
    Extrahiert einen 2D-Schnitt an Position z und korrigiert die Orientierung.
    Transformiert technische Matrix-Indizes in die radiologische Standardansicht
    (Links=Rechts, Oben=Superior) mittels Transposition und Achsenspiegelung.
    """
    s = np.transpose(vol_uint8[:, :, z])
    s = np.fliplr(s)
    s = np.flipud(s)
    return s