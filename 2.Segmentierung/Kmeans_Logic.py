import numpy as np

# Modul: K-Means Algorithmus & Initialisierung
# Implementiert unüberwachtes Clustering und Feature-Space-Logik.

def kmeans_fixed_init(X, init_centers, max_iter=40):
    """
    Führt K-Means Clustering mit fest vorgegebenen Startzentren durch.
    Dies garantiert Deterministik und Reproduzierbarkeit, im Gegensatz zur
    zufälligen Initialisierung. Nutzt Lloyd-Algorithmus (Assignment -> Update).
    """
    centers = init_centers.astype(np.float32).copy()
    K = centers.shape[0]
    labels = None
    
    for _ in range(max_iter):
        # Euklidische Distanz im N-dimensionalen Feature-Raum
        # Broadcasting: (N, 1, D) - (1, K, D) -> (N, K, D) -> Norm -> (N, K)
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)

        # Abbruch bei Konvergenz (keine Änderung der Labels)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Neuberechnung der Cluster-Zentren (Schwerpunkt)
        for k in range(K):
            pts = X[labels == k]
            if pts.shape[0] > 0:
                centers[k] = pts.mean(axis=0)
                
    return labels, centers

def labels_to_segmap(labels, idx, shape):
    """
    Rekonstruiert das 2D-Segmentierungsbild aus den flachen Vektor-Labels.
    Weist den Masken-Pixeln die berechneten Klassen-IDs (1, 2, 3) zu.
    """
    seg = np.zeros(shape, dtype=np.uint8)
    seg[idx[0], idx[1]] = (labels.astype(np.uint8) + 1)
    return seg

def compute_centers_from_quantiles(s_t1, s_fl, s_ir, mask, q_csf=0.15, q_gm=0.55, q_wm=0.90, band=10):
    """
    Berechnet datengetriebene Startzentren basierend auf T1-Statistik.
    Da T1 den höchsten anatomischen Kontrast bietet, werden Quantile (15%, 55%, 90%)
    als Anker für CSF, GM und WM genutzt. Zugehörige FLAIR/IR-Werte werden
    via Median in einem Toleranzband um den T1-Wert ermittelt.
    """
    t = s_t1[mask].astype(np.float32)
    f = s_fl[mask].astype(np.float32)
    r = s_ir[mask].astype(np.float32)

    # Fallback für leere/zu kleine Maskenbereiche
    if t.size < 50:
        return (np.array([20, 80, 80]), np.array([110, 110, 110]), np.array([235, 170, 170]))

    t_csf, t_gm, t_wm = np.quantile(t, [q_csf, q_gm, q_wm])

    # Definition der Intensitätsbänder im T1
    csf_band = t <= (t_csf + band)
    gm_band  = (t >= (t_gm - band)) & (t <= (t_gm + band))
    wm_band  = t >= (t_wm - band)

    f_med, r_med = np.median(f), np.median(r)

    def band_median(arr, band_mask, fallback):
        return float(np.median(arr[band_mask])) if np.any(band_mask) else float(fallback)

    # Konstruktion der 3D-Startvektoren
    c_csf = np.array([t_csf, band_median(f, csf_band, f_med), band_median(r, csf_band, r_med)])
    c_gm  = np.array([t_gm,  band_median(f, gm_band,  f_med), band_median(r, gm_band,  r_med)])
    c_wm  = np.array([t_wm,  band_median(f, wm_band,  f_med), band_median(r, wm_band,  r_med)])

    return c_csf, c_gm, c_wm