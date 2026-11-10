"""Analysis of the ECG, PPG and skin temperature recorded with the mouse sensor board.

Modules:
    recording   Recording: one recording loaded from its CSV file, with preprocessing methods
    filters     the filters applied to the ECG and the PPG (preprocessing)
    analysis    R waves, heart rate, breathing, averaged beats, SNR and heart-rate variability
    sessions    Session: all recordings of one isoflurane session and the dose from the notes
    stats       mean ± SD over runs and t-tests against published values
    plotting    figure style and shared figure panels

Start with walkthrough.ipynb in the main folder.
"""
from .recording import DATA, FILE, Recording, load
from .sessions import SESSIONS, Session

__all__ = ["DATA", "FILE", "Recording", "load", "SESSIONS", "Session"]
