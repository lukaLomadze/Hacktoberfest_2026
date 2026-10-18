# Data source

The four files are the original "processed" exports of the **UCI Heart Disease** dataset,
one per clinical site, downloaded unchanged from the UCI Machine Learning Repository:

| file | site | patients | sha256 |
|---|---|---|---|
| `processed.cleveland.data` | Cleveland Clinic Foundation | 303 | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| `processed.hungarian.data` | Hungarian Institute of Cardiology, Budapest | 294 | `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796` |
| `processed.switzerland.data` | University Hospitals Zurich and Basel | 123 | `834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314` |
| `processed.va.data` | V.A. Medical Center, Long Beach | 200 | `e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f` |

Source: `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.<site>.data`
(dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease), licensed CC BY 4.0.

Creators: Andras Janosi (Budapest), William Steinbrunn (Zurich), Matthias Pfisterer (Basel),
Robert Detrano (V.A. Long Beach and Cleveland Clinic Foundation). Citation: Detrano R. et al.,
"International application of a new probability algorithm for the diagnosis of coronary artery
disease", American Journal of Cardiology 64(5), 1989.

Format: 14 comma-separated values per patient (13 attributes + `num`), `?` = missing. The
column meanings are in `fedheart/config.py` (`COLUMNS`). `num > 0` means more than 50% diameter
narrowing of a major vessel; the model predicts that.
