# mPCB analysis

Simple Python scripts for the ECG and PPG recorded with the mouse sensor board.
Each script runs from top to bottom.

## Setup and running

```
py -3.13 -m pip install -r requirements.txt
py -3.13 ecg_filter_steps.py
```

The recording to analyse is `FILE` at the top of `recording.py`. Each script prints its numbers,
shows a figure and saves it as a PNG next to the scripts.

| File | What it does |
|---|---|
| `recording.py` | reads a recording: ECG in µV, PPG in nA, the sampling rates; puts the PPG on the ECG's clock |
| `filters.py` | every filter and its setting, the ECG filtering levels, the mains fit, the filter gain, the R-wave finder |
| `panels.py` | the panels both figures draw: the three steps and the averaged beat |
| `ecg_filter_steps.py` | the ECG level by level: raw, + high-pass, + 50 Hz notch, + 150 Hz notch |
| `ppg_vitals.py` | the PPG (IR): raw, after the low-pass, heart band, breathing band; vitals over time |
| `notch_differences.py` | which mains notches are worth using: each mains frequency tried on its own, the mains and the ECG it removes |
| `data_loss.py` | samples missing in the recording, and ECG samples clipped |

## How the figures are built

Each row of `ecg_filter_steps.png` and `ppg_vitals.png` is one stage of the processing. The
columns follow the build-up from the signal to the filters to the filters acting on the signal:

1. **The signal in time**, a few seconds of it.
2. **Step 1, the signal in the frequency domain** (first row). A Fourier transform splits the
   signal into sine waves; the amplitude spectrum shows how big the wave at each frequency is.
   Different parts of the signal sit at different frequencies with different amplitudes:
   breathing below 3.5 Hz, the heartbeat at its rate and at whole multiples of it (harmonics),
   the mains at 50 Hz and its multiples (marked red in the ECG figure).
   **Step 2, the filter in the frequency domain** (other rows). The filter's gain in dB at each
   frequency: 0 dB passes unchanged, -6 dB leaves half the amplitude, -20 dB a tenth. A click (a
   single 1 in a row of zeros) holds every frequency equally, so the Fourier transform of the
   filtered click is the filter's gain.
3. **Step 3, the filter acting on the signal** (all rows but the first, which has no filter).
   The power spectrum before (grey) and after the filter, in dB. In the frequency domain a filter
   multiplies every frequency by its gain, so in dB the gain is added: after = before + gain.
4. **The averaged pulse or breath** (PPG only): all pulses lined up on the ECG's R waves (or all
   breaths on their peaks), median. What happens at the same moment in every beat stays; noise
   averages away. What the filters cost the ECG's heartbeat is in `notch_differences.png`.

## The recordings

The ECG chip filters before anything is stored: a digital high-pass at 0.5 Hz and a digital
low-pass. Since 24 Sep 2026 the low-pass is at 153.6 Hz (the firmware default at 512 samples per
second, which the GUI also sends; `SET DLPF 0` would switch it off); recordings made before then
used 102.4 Hz. Since 24 Sep the cut-off is lower at other rates: 40.96 Hz at 256 and 28.35 Hz at
128 samples per second. The stored ECG (`ecg_uv`) has no notch: the GUI's notches only change
what it displays.

The analysis adds no low-pass of its own: the chip's is the only one. The mouse QRS has real
content between 100 and 150 Hz, and a second low-pass on top of the chip's would cut it (a 150 Hz
one takes 10-16 % of the QRS from recordings made since 24 Sep).

## The filters (`filters.py`)

| Filter | Setting | What it does |
|---|---|---|
| ECG high-pass | 3 Hz, Butterworth order 4 | removes baseline drift and most of the breathing |
| Mains notches | 50 and 150 Hz, Q 30 | remove the mains and its 3rd harmonic (1.7 and 5 Hz wide) |
| PPG low-pass | 32 Hz, Butterworth order 4 | removes noise above the pulse |
| PPG heart band | 4-11 Hz, Butterworth order 2 | keeps the heart-rate range (240-660 per minute) |
| PPG breathing band | 0.15-3.5 Hz, Butterworth order 2 | keeps the breathing (9-210 per minute) |

Every filter runs forwards and backwards, so nothing moves in time; it then acts twice, which is
why half the amplitude (-6 dB) is left at a cut-off. Notches at or above half the sampling rate
do not exist and are left out (at 128 samples per second only 50 Hz).

**Why only 50 and 150 Hz.** The mains has harmonics at 100, 150, 200 and 250 Hz too. A notch
removes the mains at its frequency and the ECG there with it, so it is only worth using where
the mains is much larger than that ECG. In the recordings since 24 Sep (median of 19, without
the heating pad on AC; `notch_differences.py`):

| Notch | Mains removed | ECG removed | QRS change |
|---|---|---|---|
| 50 Hz | 252 µV RMS | 25 µV RMS | -0.9 % |
| 100 Hz | 12 µV RMS | 16 µV RMS | -5.0 % |
| 150 Hz | 70 µV RMS | 18 µV RMS | -2.3 % |
| 200 Hz | 0.6 µV RMS | 0.8 µV RMS | -0.2 % |
| 250 Hz | none (the chip's low-pass removes it) | | |

The 100 Hz notch sits in the middle of the QRS's frequencies and removes more ECG than mains;
at 200 and 250 Hz there is nothing to remove. With the heating pad on AC the mains is so large
that its harmonics above 256 Hz fold back to 12, 38, 62, 112, 162 and 212 Hz, where no notch at
a multiple of 50 Hz reaches; no notch can bring that ECG back.

## How the numbers are worked out

**R waves.** The R wave is a sharp spike, so most of it sits at 20-90 Hz. In that band, with the
mains notched out, peaks taller than 0.3 x the typical tall peak are R waves; each mark is then
moved to the top of the ECG nearby. After each mark the finder waits 0.6 x the typical time
between beats before it takes the next one, so a tall T wave (about 70 ms after R) is not
counted as a second beat. The typical time between beats comes from the autocorrelation of the
band's energy, searched from 80 to 300 ms (750 to 200 beats per minute) so breathing is not
taken for it. Shifted by two beats it looks alike too, so one beat is the shortest delay at
least 0.8 times as alike as the best one; if the best is less than half as alike as delay 0, no
beat repeat is clear and the wait is 80 ms. The wait and the height threshold are worked out for
every 30 s on its own, because the heart rate can change by half within one recording. Movement
can still add false beats. Heart rate = 60 / time between two R waves. The scripts print in how
many 30 s pieces the beat repeats clearly. In a recording with no heartbeat (open leads, copper
tape, mains swamping the ECG) it is 0, and the heart rate printed is noise. Human recordings
also show 0, because only mouse rates are searched; their R waves are large enough to be found
anyway.

**Averaged beat.** All beats lined up on their R waves, median, from 100 ms before to 100 ms
after R. QRS = highest minus lowest point from 15 ms before to 25 ms after R.

**PPG on the ECG's clock.** The PPG chip's clock runs about 0.2 % fast, so its samples drift
against the ECG. `recording.py` counts the PPG samples per ECG second and uses that as the PPG
sampling rate. In a complete recording the PPG has 0.15-0.35 % more samples than its setting
says; anything else means samples were lost, and the script warns that PPG-to-ECG timing is not
reliable.

**Vitals, per 30 s window** (`ppg_vitals.py`):
- Heart rate from the ECG, and from the PPG as the strongest frequency in the heart band (Welch).
  In mice the heart band also holds breathing harmonics, so the PPG value can miss the heart.
- Breathing rate from the autocorrelation of the breathing band: how much the signal looks like
  itself when shifted by a delay. It looks most alike when shifted by exactly one breath, also
  when each breath has two humps, where counting peaks would count two breaths. The likeness is
  averaged over the part that overlaps, and only its peaks count (at the shortest delays it is
  still falling from delay 0). Shifted by two breaths it looks alike too, so one breath is the
  shortest delay at least 0.8 times as alike as the best one. A window gets a value only if the
  best peak is at least half as alike as delay 0; otherwise its breathing is not clear (a moving
  animal, or breathing faster than the band's 210 per minute).
- Perfusion index = the size (peak to peak) of the pulse averaged on the R waves, in % of the
  light level. Taken from the averaged pulse because the heart band alone holds breathing too.

**ECG filtering levels.** Each level adds one filter to the level before: raw, + 3 Hz high-pass,
+ 50 Hz notch, + 150 Hz notch. For finding the R waves, every mains frequency is notched out.

**Notch differences.** Each mains frequency (50, 100, 150, 200, 250 Hz) is tried as a notch on
its own. To show only the ECG a notch removes, it is tried on the ECG with the mains fitted out:
in each second, the mains waves that best match the ECG are found and subtracted, so what a
notch then removes can only be ECG. Each row of `notch_differences.png` shows the ECG after the
notch with the ECG it removed beside it, the removed ECG enlarged, the averaged beat before and
after, and the recording's power spectrum close to the notch before and after (a tall line
there is mains). The script prints how much mains and how much ECG each notch removes. The fit
itself removes a little ECG too (about 1-2 % of the QRS), so the losses shown are slightly
smaller than the true ones.

**Data loss.** Gaps: a step between two time stamps longer than 1.5 samples means samples are
missing there. Loss without a gap: lost samples make the signal look sped up, so the mains
appears above 50 Hz (one sample lost in 91 moves it to 50.56 Hz); the lost share is
1 - 50 / that frequency. Clipping: the share of ECG samples at the end of the chip's range
(-131072 to 131071, `ecg_raw`), where the ECG is cut off.

## Which recordings are used

- Recordings without a settings line (the first line starting with `#`) are not used.
- The PPG is used only when the settings line gives its range (`adc_range`), the IR level is at
  least 150 nA (below that the sensor is not on the skin) and the IR is not clipped at the top of
  the range for most of the recording.
- SpO2 is not calculated: its constants A and B have no calibration for this sensor.

## Sources

- Bhadouria VS et al. (2025) Signal Image Video Process 19:231, p. 5: mouse heart rate as the
  highest Welch peak in 4-11 Hz.
- Oliveira RG et al. (2022) Sensors 22:4264, p. 6: mouse breathing band 0.5-3.5 Hz (heartbeat
  band 5.5-8 Hz, too narrow for these mice).
- Serri Mazandarani M (2025) Photoplethysmography systems for vital signs monitoring in humans and
  small animal models, thesis, Université Laval, pp. 70 and 73: PPG low-pass 32 Hz (used there on
  human finger PPG; no mouse value is given).
