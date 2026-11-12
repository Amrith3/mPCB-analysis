# mPCB analysis

Python code for the ECG, PPG and skin temperature recorded with the mouse sensor board. It reads the
recordings, filters them, finds the heartbeats and breaths, checks the signal quality, follows the vital
signs through the isoflurane doses and makes the figures of the results report.

## Folders

| Folder or file | What it is |
|---|---|
| `walkthrough.ipynb` | Start here: the whole analysis step by step, with explanations and figures |
| `mpcb/` | The shared code that reads, filters and measures the recordings; the notebook and every script use it |
| `scripts/` | Scripts for the signal analysis: filter steps, mains notches, PPG, signal quality, PPG against ECG, checks |
| `isoflurane/` | Scripts for the isoflurane dose analysis, with its own README |
| `report/` | Scripts for the graphs and t-tests of the results report |
| `data/` | The recordings, one zip per session, unzipped the first time they are used |
| `outputs/` | Every figure and table the scripts make; created when a script runs, not kept in git |
| `METHODS.md` | How each filter and measurement works, how the methods were checked, and the sources |
| `requirements.txt` | The package versions the results were made with |

## Running it

Python 3.13 and VS Code with its Python and Jupyter extensions are needed.

1. Open this folder in VS Code (File, Open Folder).
2. Open a terminal (Terminal, New Terminal) and install the packages once:

   ```
   py -3.13 -m pip install -r requirements.txt
   ```

3. Open `walkthrough.ipynb`, choose Python 3.13 as the kernel (top right) and click Run All.

A script is run in the terminal from this folder, for example:

```
py -3.13 scripts/checks.py
```

`checks.py` tries every method on made-up signals with a known answer and ends with "all checks passed".

| Script | What it saves in `outputs/` | Recordings |
|---|---|---|
| `scripts/ecg_filter_steps.py` | `ecg_filter_steps.png`: the ECG after each filter step | `FILE` |
| `scripts/notch_differences.py` | `notch_differences.png`: the mains and the ECG each notch removes | `FILE` |
| `scripts/ecg_frequency_map.py` | `ecg_frequency_ranges.png` (bars on one frequency axis) and `ecg_frequency_map.png` (low cut against high cut): the ECG's frequency bands, human and mouse, against what a recorder keeps | the 3 dose runs, every session |
| `scripts/ppg_vitals.py` | `ppg_vitals.png`: the PPG after each filter step, and the vitals every 30 s | `FILE` |
| `scripts/quality.py` | `quality.csv`: the signal quality of every recording | all 21 |
| `scripts/hr_agreement.py` | `hr_agreement.png`: PPG against ECG heart rate (Bland-Altman) | 14: usable PPG and a clear ECG |
| `scripts/checks.py` | nothing, it prints its results | made-up signals, 22 Oct 14:54:30 for real noise |
| `isoflurane/timeline.py` | `isoflurane/timeline.png`: one session over time | 26 Oct |
| `isoflurane/doses.py` | `isoflurane/doses.png`: averaged beat, pulse and breathing at each dose | 26 Oct |
| `isoflurane/table.py` | `isoflurane/*.csv`: the values at each dose | 3 runs, 14 recordings |
| `report/figures.py` | `report/*.png`: the graphs of the results report | `FILE`, 26 Oct, the 3 dose runs |
| `report/statistics.py` | `report/t_tests.csv`: t-tests against published values (run `isoflurane/table.py` first) | the 3 dose runs |

`FILE`, at the top of `mpcb/recording.py`, is the recording the single-recording scripts use: 22 Oct,
14:54:30 (Mouse 1, 30 g). In the notebook any recording can be read with
`Recording("data/.../file.csv")`.

## The mpcb package

| Module | What it holds |
|---|---|
| `mpcb/recording.py` | `Recording`: one recording read from its CSV file (ECG in µV, PPG in nA, skin temperature, notes), with methods to filter it, measure it and check its quality |
| `mpcb/filters.py` | every filter and its setting |
| `mpcb/analysis.py` | R waves, heart rate from the ECG and the PPG, breathing rate, averaged beat and pulse, perfusion index, SNR, HRV |
| `mpcb/sessions.py` | `Session`: all recordings of one isoflurane session on one clock, the dose from the notes, the vitals every 10 s |
| `mpcb/stats.py` | mean ± SD, median (IQR), t-tests against published values |
| `mpcb/plotting.py` | the figure style and the panels of the filter figures |

Every function starts with a description of what it does and how; `help()` shows it, for example
`help(filters.ecg_highpass)`.

## Recordings

The recordings since 24 Sep 2026: 8, 20, 22 and 26 Oct (22 Oct holds two mice, 30 g and 25 g). Which
recordings and signals are used, and why, is in `METHODS.md`.
