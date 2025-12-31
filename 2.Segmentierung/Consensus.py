import numpy as np
from scipy import stats

# Modul: Volumetrisches Consensus Voting
# Reduziert Bildrauschen durch Analyse benachbarter Slices.

def apply_consensus_voting(segmentation_stack):
    """
    Berechnet den Pixel-weisen Modalwert (Mehrheitsentscheid) über einen Stack von Slices.
    Eliminiert inkonsistente Klassifikationen (Salt-and-Pepper Rauschen), 
    indem räumliche Kohärenz entlang der Z-Achse erzwungen wird.
    """
    # stats.mode liefert den häufigsten Wert entlang axis=0 (Z-Achse)
    mode_result, _ = stats.mode(segmentation_stack, axis=0, keepdims=True)
    return mode_result[0]