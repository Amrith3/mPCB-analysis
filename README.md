# mPCB analysis

Python scripts for the ECG and PPG recorded with the mouse sensor board.

```
py -3.13 -m pip install -r requirements.txt
py -3.13 checks.py
py -3.13 ecg_filter_steps.py
```

`requirements.txt` holds the versions the results were made with. `checks.py` tries the methods on
made-up signals with a known answer and ends with "all checks passed". The recordings since 24 Sep
2026 are in `data/`, one zip per session, unzipped on first use (22 Oct holds two mice: 30 g and
25 g). Run the scripts from this folder; figures (PNG, PDF) and tables (CSV) are saved here.

## Files

| File | Output | Recordings |
|---|---|---|
| `recording.py` | ECG in µV, PPG in nA, sampling rates; the PPG on the ECG's clock | `FILE` = 22 Oct 14:54:30 |
| `filters.py` | every filter and setting, R waves and R times, SNR, HRV and its floor | |
| `panels.py` | the panels of the step figures; the style of the other figures | |
| `ecg_filter_steps.py` | the ECG stage by stage (`ecg_filter_steps.png`) | `FILE` |
| `notch_differences.py` | the mains and ECG each notch removes (`notch_differences.png`) | `FILE` |
| `ppg_vitals.py` | the PPG stage by stage; vitals per 30 s (`ppg_vitals.png`) | `FILE` |
| `data_loss.py` | lost and clipped samples (printed) | `FILE` |
| `quality.py` | signal quality per recording (`quality.csv`) | all 21 |
| `hr_agreement.py` | PPG against ECG heart rate, Bland-Altman (`hr_agreement.pdf`) | 14: usable PPG, clear ECG, at least 20 s |
| `checks.py` | the methods on made-up signals (printed) | 22 Oct 14:54:30 for real noise |
| `isoflurane/` | heart rate, breathing, ECG beat and pulse against the dose; see its README | 3 runs, 14 recordings |

## Filters

| Filter | Setting | Purpose |
|---|---|---|
| ECG high-pass | 3 Hz, Butterworth 4 | removes drift and most of the breathing |
| Mains notches | 50 and 150 Hz, Q 30 | the only mains lines worth removing (below) |
| ECG low-pass | none of its own | the chip's 153.6 Hz is enough; a 150 Hz one would take up to 15 % of the QRS (median 7 %) |
| PPG low-pass | 32 Hz, Butterworth 4 | removes noise above the pulse |
| PPG heart band | 4-11 Hz, Butterworth 2 | 240-660 per minute |
| PPG breathing band | 0.15-3.5 Hz, Butterworth 2 | 9-210 per minute |

All filters run forwards and backwards: nothing moves in time, and -6 dB is left at a cut-off.
Notches at or above half the sampling rate are left out.

### Mains notches

A notch removes the ECG at its frequency too, so it is worth using only where
the mains is much larger. Median of the 19 recordings since 24 Sep without the heating pad
(`notch_differences.py`; the ECG removed is measured with the mains fitted out, which itself costs
1-2 % of the QRS):

| Notch | Mains removed | ECG removed | QRS change | Used |
|---|---|---|---|---|
| 50 Hz | 253 µV RMS | 8 µV RMS | -0.9 % | yes |
| 100 Hz | 12 µV RMS | 16 µV RMS | -4.9 % | no: removes more ECG than mains |
| 150 Hz | 72 µV RMS | 18 µV RMS | -2.7 % | yes |
| 200 Hz | 0.8 µV RMS | 0.8 µV RMS | -0.2 % | no: nothing to remove |
| 250 Hz | none | | | no: the chip's low-pass removes it |

With the heating pad on AC, mains above 256 Hz folds back to 12, 38, 62, 112, 162 and 212 Hz,
where no notch reaches.

## Methods

| Measure | How |
|---|---|
| R waves | in the 20-90 Hz band with all mains notched: peaks above 0.3 x the typical tall peak, at least 0.6 x the beat interval apart (skips the T wave); the beat interval from the band's autocorrelation, 80-300 ms; worked out per 30 s, as the rate changes by half within a recording |
| Clear beat | the beat interval repeats clearly (autocorrelation peak at least 0.5); 0 pieces means no heartbeat (open leads, mains, human recordings) and the rate printed is noise |
| R peak | the earliest peak at least 0.6 x the tallest from 20 ms before to 5 ms after, on the high-passed ECG, so a J wave taller than R is not taken |
| R times | each QRS matched to the averaged beat of its 30 s and read between samples, within 0.1 ms; a mark on J (40 % of beats at 4 % isoflurane) no longer moves it |
| Heart rate | 60 / R-R |
| Averaged beat | median of the beats lined up on R, -100 to +100 ms; QRS = highest minus lowest point from -15 to +25 ms |
| HRV | SDNN and RMSSD of the normal R-R in 30 s; normal = within 5 robust SDs of the median, at least 2 samples and at most 20 % from it |
| HRV floor | what a perfectly regular heart reads with the recording's noise: RMSSD 0.11-0.39 ms, SDNN 0.07-0.22 ms here; a value near it is mostly noise |
| SNR | beats lined up on their own R times (between samples); their average is the signal, what each beat differs from it the noise; 10 x log10 of the power ratio |
| PPG heart rate | the strongest Welch peak in the heart band; breathing harmonics can take its place |
| Breathing rate | the shortest autocorrelation peak at least 0.8 x the best, clear if the best is at least 0.5; a breath with two humps counts once |
| Perfusion index | peak to peak of the pulse averaged on the R waves, in % of the light level |
| PPG clock | PPG samples counted per ECG second; the PPG chip runs 0.15-0.35 % fast, anything else means lost samples |
| Data loss | a time step longer than 1.5 samples; without a gap, lost samples move the mains above 50 Hz (1 in 91 gives 50.56 Hz) |
| Clipping | ECG samples at the end of the chip's range (±131071) |

### HRV

SDNN holds all of the variability, a drifting heart rate too; RMSSD the beat-to-beat
part. Both are smaller over shorter stretches and at faster heart rates, so only 30 s values are
compared, and part of a change with the dose is the change of heart rate. `quality.py` gives the
median (IQR) of each recording's 30 s pieces; per dose they are in `isoflurane/`.

### SNR

Lined up only to whole samples (1.95 ms), the SNR reads 1.5-6.6 dB too low. What the
recordings reach and what is enough (`checks.py` with real noise made larger, `hr_agreement.py`):

| ECG | SNR | Beats found | Extra beats |
|---|---|---|---|
| chip's own noise, 0.82 µV RMS | would allow 35-45 dB | | |
| recordings since 20 Oct | 6-15 dB | | |
| 8 Oct, heating pad on AC | below 0 dB | | |
| real noise of 22 Oct, as recorded | 13 dB | 99.9 % | 0.4 % |
| the same noise x2 | 7 dB | 97 % | 3.5 % |
| the same noise x3 | 4 dB | 95 % | 6.8 % |

The heart rate stays right in all three (the R-R rules drop the extra beats). The PPG heart
rate is within 5 % of the ECG's in 117 of 118 pieces with a PPG SNR above -5 dB, and in 51 of 88
below.

## Figures

Each row of `ecg_filter_steps.png` and `ppg_vitals.png` is one stage of the processing:

| Column | Shows |
|---|---|
| 1 | a few seconds of the signal |
| 2 | first row: the amplitude spectrum, mains marked red; other rows: the filter's gain in dB (0 passes, -6 half, -20 a tenth), the spectrum of a filtered click |
| 3 | the power spectrum before (grey) and after the filter; in dB, after = before + gain |
| 4 | PPG only: the pulse averaged on the R waves, or the breath on its peaks |

## Recordings used

| Used | When |
|---|---|
| the recording | it has a settings line (first line, starting with `#`) |
| the PPG | the settings line gives its range (`adc_range`), IR at least 150 nA (on the skin), not clipped for most of the recording |
| SpO2 | never: its constants A and B have no calibration for this sensor |

## Sources

- Bhadouria VS et al. (2025) Signal Image Video Process 19:231, p. 5: mouse heart rate as the
  highest Welch peak in 4-11 Hz.
- Oliveira RG et al. (2022) Sensors 22:4264, p. 6: mouse breathing band 0.5-3.5 Hz.
- Serri Mazandarani M (2025) Photoplethysmography systems for vital signs monitoring in humans and
  small animal models, thesis, Université Laval: PPG low-pass 32 Hz on human finger PPG (pp. 70 and
  73); PPG SNR as the power at the pulse rate and its harmonics against the noise (pp. 67-68);
  28.5 dB for a heart rate within 1 per minute (p. 24); a clear mouse PPG above 10 dB (p. 89).
- Analog Devices, MAX30003 datasheet, p. 3: input-referred noise 0.82 µV RMS (0.05-150 Hz).
- Shaffer F, Ginsberg JP (2017) Front Public Health 5:258: SDNN and RMSSD (pp. 2 and 4); a faster
  heart rate lowers the HRV (p. 8); shorter recordings give smaller values (p. 13).
