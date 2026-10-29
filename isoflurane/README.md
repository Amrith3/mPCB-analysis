# Isoflurane doses

How heart rate, breathing, the ECG beat, the PPG pulse and skin temperature change with the
isoflurane dose. It uses the recordings in `../data` (unzipped on first use) and the filters of the
main folder.

```
py -3.13 isoflurane/timeline.py
py -3.13 isoflurane/doses.py
py -3.13 isoflurane/table.py
```

| File | What it does |
|---|---|
| `sessions.py` | each session's recordings on one clock, the dose from the notes, the vitals every 10 s, the averaged beat at each dose and its wave boundaries |
| `timeline.py` | one session over time: dose, heart rate, breathing rate, perfusion index, skin temperature (`timeline.pdf`) |
| `doses.py` | one column per dose: the averaged ECG beat, the averaged PPG pulse, 20 s of breathing, skin temperature (`doses.pdf`) |
| `table.py` | per dose: the vitals as mean ± SD and CV, and the ECG intervals and wave sizes of the averaged beat (`table.csv`) |

Change `SESSION` at the top of `timeline.py` or `doses.py` for another session.

## How the numbers are worked out

- **Dose**: read from the notes typed during the recording: the number after "to" or "at"
  ("iso to 4" is 4 %). A note about isoflurane without a number ("gradually decreasing") leaves the
  dose unknown from then on. The dose is the vaporizer setting.
- **Every 10 s**: heart rate from the R waves (60 / mean R-R, R-R outside 80-300 ms or more than
  30 % from the median left out); breathing rate and perfusion index from 30 s of PPG around it;
  skin temperature. Where the ECG's noise is more than 3 x, or its mains more than 5 x, the
  session's usual, there is no heart rate (movement, the heating pad on AC). The perfusion index
  counts only where the PPG's heart rate is within 5 % of the ECG's. A breathing rate more than
  30 % from the median of the minute around it is left out (two breaths taken for one).
- **At each dose**: the last 2 min recorded at it. Each dose lasted about 5 min, and heart rate
  was often still changing at the end, so these are values after about 5 min, not a steady state.
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

- Heart rate agrees with the ECG chip's own R-R detector where that detector finds every beat;
  where they differ, the chip has missed beats and the PPG agrees with this heart rate.
- Breathing rate agrees with the strongest frequency of the breathing band (Welch) within 10 % in
  89-100 % of the windows.
- The wave boundaries are found within 0.7 ms on a made-up beat with known boundaries.

## Sources

- Constantinides C, Mean R, Janssen BJ (2011) ILAR J 52:e21-e31: steady-state values at each dose,
  mean ± SD and coefficient of variation (pp. e23-e24).
- Tsukamoto A et al. (2015) Exp Anim 64:57-64: vital signs every 5 min, mean ± SD, coefficient of
  variation (pp. 59-61).
- Merentie M et al. (2015) Physiol Rep 3:e12639: averaged beat and wave boundaries (p. 3); PQ above
  60 ms is a first-degree AV block in mice (p. 11).
- Boukens BJ et al. (2014) J Physiol 592:4613-4626: in mice the ventricles start at the top of the
  septum, so the QRS starts with R (p. 4616).
