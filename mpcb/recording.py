"""One recording of the GUI as an object: the Recording class.

    rec = Recording("data/mouse_isoflurane_20261022/mouse_30g/mice_anest_20261022_145430.csv")
    rec.ecg                      # ECG in µV, as recorded
    rec.ecg_filtered()           # after the 3 Hz high-pass and the 50 and 150 Hz notches
    rec.heart_rate()             # heart rate every 10 s
    rec.breaths()                # single breaths from the PPG

The CSV file of the GUI holds one row per sample: ECG rows (512 per second), PPG rows (200 or
50 per second) and notes typed during the recording. Its first line holds the settings.
"""
import os
import zipfile

import numpy as np
import pandas as pd

from . import analysis, filters

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FILE = os.path.join(DATA, "mouse_isoflurane_20261022", "mouse_30g", "mice_anest_20261022_145430.csv")


def unzip(session):
    """Unzip data/<session>.zip into data/ the first time it is needed."""
    if not os.path.isdir(os.path.join(DATA, session)):
        zipfile.ZipFile(os.path.join(DATA, session + ".zip")).extractall(DATA)


class Recording:
    """One recording: ECG, PPG and skin temperature, their sampling rates, and the notes.

    Attributes after loading:
        file, settings      the CSV file and its settings line (dict)
        fs_ecg, t_ecg       ECG sampling rate (Hz) and sample times (s)
        ecg, ecg_raw        ECG in µV, and the chip's counts
        notes               [(time in s, text)] typed in the GUI
        t_temp, temp_c      skin temperature times (s) and values (°C)
        has_ppg             whether the PPG can be used (ppg_problem says why not)
        fs_ppg, t_ppg       PPG sampling rate (Hz) and times (s), on the ECG's clock
        ir_nA, red_nA       infrared and red light in nA
    """

    def __init__(self, file=FILE):
        """Read the recording. It is unzipped from data/ first if needed.

        How the units are made: the ECG in µV is stored by the GUI. The PPG counts are turned into
        nA with the light range of the settings line (nA = count x adc_range / 262143). The PPG is
        not used if the settings line gives no range, if the infrared light is below 150 nA (the
        sensor was not on the skin) or if it sits at the top of its range for most of the recording.

        The PPG chip's clock runs about 0.2 % fast, so its own times drift against the ECG. The PPG
        sampling rate is therefore taken as the number of PPG samples per second of ECG, which puts
        the PPG on the ECG's clock. A complete recording has 0.15-0.35 % more PPG samples than its
        setting says; anything else means samples were lost, and a warning is printed.
        """
        inside = os.path.relpath(os.path.abspath(file), DATA)
        if not os.path.exists(file) and not inside.startswith(".."):
            unzip(inside.split(os.sep)[0])
        self.file = file
        with open(file) as f:
            first_line = f.readline()
        self.settings = {}
        for part in first_line.split():
            if "=" in part:
                name, value = part.split("=")
                self.settings[name] = value
        if "ecg_rate" not in self.settings:
            raise SystemExit("This recording has no settings line, so it is not used.")

        data = pd.read_csv(file, comment="#", low_memory=False)
        ecg = data[data["ecg_uv"].notna()]
        self.fs_ecg = float(self.settings["ecg_rate"].rstrip("sp"))
        self.t_ecg = ecg["time_s"].to_numpy()
        self.ecg = ecg["ecg_uv"].to_numpy()
        self.ecg_raw = ecg["ecg_raw"].to_numpy()
        seconds = len(self.ecg) / self.fs_ecg
        print("ECG:", len(self.ecg), "samples at", self.fs_ecg, "per second")
        self.notes = []
        if "note" in data:                       # older recordings have no note column
            notes = data[data["note"].notna()]
            self.notes = list(zip(notes["time_s"], notes["note"].astype(str)))
        temp = data[data["temp_c"].notna()]
        self.t_temp = temp["time_s"].to_numpy()
        self.temp_c = temp["temp_c"].to_numpy()

        self.has_ppg = False
        self.ppg_problem = None
        adc_range = self.settings.get("adc_range", "unknown")
        if not adc_range.isdigit():
            self.ppg_problem = "the settings line gives no PPG range"
            print("PPG not used:", self.ppg_problem)
            return
        ppg = data[data["ir"].notna()]
        ir_nA = ppg["ir"].to_numpy() * float(adc_range) / 262143
        if ir_nA.mean() < 150:
            self.ppg_problem = "IR is %d nA, the sensor was not on the skin" % round(ir_nA.mean())
            print("PPG not used:", self.ppg_problem)
            return
        if (ir_nA >= 0.99 * float(adc_range)).mean() > 0.5:
            self.ppg_problem = "IR is clipped at the top of its range (too much light)"
            print("PPG not used:", self.ppg_problem)
            return
        self.has_ppg = True
        self.ir_nA = ir_nA
        self.red_nA = ppg["red"].to_numpy() * float(adc_range) / 262143

        written = float(self.settings["ppg_sr"].rstrip("sp"))
        counted = len(ir_nA) / seconds
        fast = counted / written
        self.fs_ppg = counted if abs(fast - 1) < 0.006 else written
        if fast < 1.0015 or fast > 1.0035:
            print("warning: the PPG has", round(100 * (fast - 1), 2), "% more samples than its setting says"
                  " (0.15 to 0.35 % in a complete recording), so samples were lost: PPG-to-ECG timing is not reliable")
        self.t_ppg = ppg["time_s"].to_numpy() * written / self.fs_ppg
        print("PPG:", len(ir_nA), "samples at", round(self.fs_ppg, 2), "per second")

    def __repr__(self):
        ppg = "PPG %.2f per second" % self.fs_ppg if self.has_ppg else "no PPG"
        return "Recording(%s: %.0f s, ECG %g per second, %s)" % (
            os.path.basename(self.file), self.t_ecg[-1], self.fs_ecg, ppg)

    # preprocessing ------------------------------------------------------------------------------

    def ecg_filtered(self):
        """The ECG after the 3 Hz high-pass and the 50 and 150 Hz notches (filters.py)."""
        return filters.notches(filters.ecg_highpass(self.ecg, self.fs_ecg), self.fs_ecg)

    def ecg_without_mains(self):
        """The ECG after the 3 Hz high-pass with the mains fitted out instead of notched, for
        measuring small waves on the averaged beat (filters.mains_fitted_out)."""
        return filters.ecg_highpass(filters.mains_fitted_out(self.ecg, self.fs_ecg), self.fs_ecg)

    def ppg_filtered(self):
        """The infrared PPG after the 32 Hz low-pass (filters.ppg_lowpass)."""
        self._need_ppg()
        return filters.ppg_lowpass(self.ir_nA, self.fs_ppg)

    def heart_band(self):
        """The infrared PPG in its heart-rate band, 4-11 Hz (filters.heart_band)."""
        return filters.heart_band(self.ppg_filtered(), self.fs_ppg)

    def breathing_band(self):
        """The infrared PPG in its breathing band, 0.15-3.5 Hz (filters.breathing_band)."""
        return filters.breathing_band(self.ppg_filtered(), self.fs_ppg)

    # analysis -----------------------------------------------------------------------------------

    def r_waves(self, pieces_out=None):
        """Sample numbers of the R waves (analysis.r_waves)."""
        return analysis.r_waves(self.ecg, self.fs_ecg, pieces_out)

    def r_times(self, r=None):
        """R-wave times in s, between samples (analysis.r_times); r: R-wave samples if already found."""
        if r is None:
            r = self.r_waves()
        return analysis.r_times(self.ecg, self.t_ecg, self.fs_ecg, r)

    def heart_rate(self, step_s=10, r=None):
        """Times (s, middle of each window) and heart rate (beats per minute) every step_s seconds,
        from the R-R intervals in each window (analysis.window_heart_rate)."""
        if r is None:
            r = self.r_waves()
        r_t = self.t_ecg[r]
        rr, rr_t = np.diff(r_t), r_t[1:]
        starts = np.arange(0, self.t_ecg[-1] - step_s, step_s)
        rate = [analysis.window_heart_rate(rr[(rr_t >= s) & (rr_t < s + step_s)]) for s in starts]
        return starts + step_s / 2, np.array(rate)

    def breathing_rate(self, start=None, end=None):
        """Breaths per minute between start and end (s) of the recording, from the breathing band
        (analysis.breathing_rate); the whole recording if they are not given."""
        band = self.breathing_band()
        inside = np.ones(len(band), bool)
        if start is not None:
            inside &= self.t_ppg >= start
        if end is not None:
            inside &= self.t_ppg < end
        return analysis.breathing_rate(band[inside], self.fs_ppg)

    def breaths(self):
        """Times (s) and rates (breaths per minute) of the single breaths (analysis.breaths)."""
        return analysis.breaths(self.breathing_band(), self.t_ppg, self.fs_ppg)

    def averaged_beat(self, half_ms=100, r=None):
        """Times (ms from R) and the median ECG beat (µV) around the R waves, mains fitted out."""
        if r is None:
            r = self.r_waves()
        half = int(half_ms * self.fs_ecg / 1000)
        beat = analysis.averaged(self.ecg_without_mains(), r, half)
        return 1000 * np.arange(-half, half) / self.fs_ecg, beat

    def averaged_pulse(self, half_ms=150, r=None):
        """Times (ms from R) and the median PPG heart-band pulse (nA) around the ECG's R waves."""
        if r is None:
            r = self.r_waves()
        half = int(half_ms * self.fs_ppg / 1000)
        pulses = np.searchsorted(self.t_ppg, self.t_ecg[r])
        pulse = analysis.averaged(self.heart_band(), pulses, half)
        return 1000 * np.arange(-half, half) / self.fs_ppg, pulse

    def _need_ppg(self):
        if not self.has_ppg:
            raise ValueError("No usable PPG in this recording: " + str(self.ppg_problem))


def load(file=FILE):
    """The same as Recording(file)."""
    return Recording(file)
