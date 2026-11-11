# Isoflurane doses

How heart rate, breathing, the ECG beat, the PPG pulse and skin temperature change with the
isoflurane dose. It uses the recordings in `../data` (unzipped on first use) and the code in `../mpcb`. The
figures and tables are saved in `../outputs/isoflurane/`.

```
py -3.13 isoflurane/timeline.py
py -3.13 isoflurane/doses.py
py -3.13 isoflurane/table.py
```

| File | What it does |
|---|---|
| `../mpcb/sessions.py` | each session's recordings on one clock, the dose from the notes, the vitals every 10 s, the averaged beat at each dose and its wave boundaries |
| `timeline.py` | one session over time: dose, heart rate, breathing rate, perfusion index, skin temperature (`timeline.pdf`) |
| `doses.py` | one column per dose: the averaged ECG beat, the averaged PPG pulse, 20 s of breathing, skin temperature (`doses.pdf`) |
| `table.py` | per dose of each run: the vitals as mean ± SD (`vitals.csv`), the intervals of the averaged beat (`ecg.csv`), the signal-to-noise ratios and heart-rate variability as median (IQR), with the floor of the heart-rate variability (`snr_hrv.csv`), and all of it with the CV and the wave sizes (`table.csv`) |

Change `SESSION` at the top of `timeline.py` or `doses.py` for another session.

## How the numbers are worked out

- **Dose**: read from the notes typed during the recording: the number after "to" or "at"
  ("iso to 4" is 4 %). A note about isoflurane without a number ("gradually decreasing") leaves the
  dose unknown from then on. The dose is the vaporizer setting.
- **Every 10 s**: heart rate from the R waves (60 / mean R-R, R-R outside 80-300 ms or more than
  30 % from the median left out); skin temperature; from the 30 s around it, breathing rate and
  perfusion index (PPG), the signal-to-noise ratio of the ECG and of the PPG, SDNN and RMSSD (see
  `../METHODS.md`). Where the ECG's noise is more than 3 x, or its mains more than 5 x, the
  session's usual, there is no heart rate, SNR or HRV (movement, the heating pad on AC). The
  perfusion index counts only where the PPG's heart rate is within 5 % of the ECG's. A breathing
  rate more than 30 % from the median of the minute around it is left out (two breaths taken for
  one).
- **At each dose**: the last 2 min recorded at it. The steps at 1, 2 and 4 % lasted 2-5.5 min (the
  1 % step of the 30 g mouse on 22 Oct only 2.1 min, so its last 2 min are almost the whole step),
  and heart rate was often still changing at the end, so these are values after a few minutes, not
  a steady state.
  In the tables, ± is the SD of the 10 s values over those 2 min, within one animal, as the
  literature reports vitals (Constantinides et al. 2011, Tsukamoto et al. 2015); the SNR and HRV
  are given as median (IQR) of the same values, since a few noisy windows skew them; next to the
  HRV, its floor: what a perfectly regular heart would read with the noise of those 2 min (`../METHODS.md`). The
  perfusion index in `vitals.csv` is in % of the run's 1 % value: its size depends on how firmly
  the sensor sits, so only its change within a run is compared.
- **Averaged beat**: the ECG with the mains fitted out (a notch rings after the QRS), the beats
  lined up so their QRS matches the first dose's best (isoflurane makes the J wave taller than R),
  median. Wave boundaries as in Merentie et al. (2015): where the slope falls below 0.15 x that
  wave's steepest slope; QRS from the start of the R upstroke to halfway between the S and J peaks;
  sizes from the level at P onset. A mouse QRS starts with R (no Q wave); the dip before R is the
  ECG chip's filter ringing. A time is given only where the odd and the even beats give the same
  within 2 ms, and the P wave not when R-R is within 1.5 ms of a whole number of mains cycles
  (20 ms), since the mains left then does not average away. QT is left out: the end of the T wave
  is not reliable here.

## Checks

See the Checks section of `../METHODS.md`.

## Sources

- Constantinides C, Mean R, Janssen BJ (2011) ILAR J 52:e21-e31: steady-state values at each dose,
  mean ± SD and coefficient of variation (pp. e23-e24).
- Tsukamoto A et al. (2015) Exp Anim 64:57-64: vital signs every 5 min, mean ± SD, coefficient of
  variation (pp. 59-61).
- Merentie M et al. (2015) Physiol Rep 3:e12639: averaged beat and wave boundaries (p. 3); PQ above
  60 ms is a first-degree AV block in mice (p. 11).
- Boukens BJ et al. (2014) J Physiol 592:4613-4626: in mice the ventricles start at the top of the
  septum, so the QRS starts with R (p. 4616).
