# UCI HAR — Processed

Canonical HDF5 build of the **UCI HAR (Human Activity Recognition Using Smartphones)** dataset
for the TTL-TSFM pretraining corpus. See `docs/DATA.md` for the corpus-wide spec; this README
is the dataset-local provenance record.

---

## 1. Provenance

| Field | Value |
|---|---|
| **Source** | UCI Machine Learning Repository, dataset 240 |
| **URL** | https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip |
| **Original publication** | Anguita et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN 2013 |
| **License** | CC BY 4.0 (UCI ML Repository) |
| **Sensor** | Samsung Galaxy S II smartphone — accelerometer + gyroscope |
| **Placement** | Waist (smartphone clipped to subject's belt) |
| **Native sampling rate** | 50 Hz |
| **Subjects** | 30 (ages 19–48) |
| **Activities** | 6: walking, walking_upstairs, walking_downstairs, sitting, standing, laying |
| **Original protocol** | Each subject performed each activity twice; data pre-windowed by authors into 2.56 s frames at 50 % overlap (128 samples / 64 stride) |
| **Parser** | `src/datasets/uci_har.py` (class `UCIHARParser`, base `IMUDataset`) |
| **Processed date** | 2026-03-04 |
| **Validated date** | 2026-04-11 |
| **Schema version** | 1.0 |

---

## 2. Files

```
data/processed/uci_har/
├── windows.h5      # canonical HDF5 (this dataset)
├── metadata.json   # dataset card — emitted by parser
├── stats.json      # validator output — emitted by dataset-validator
└── README.md       # this file
```

---

## 3. HDF5 Contents

Path: `windows.h5` — chunks `(256, 512, 6)`, LZF compression, libver `latest`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `/windows` | (5125, 512, 6) | float32 | Fixed-range scaled to [−1, 1] (see §6) |
| `/windows_raw` | (5125, 512, 6) | float32 | Physical units (acc in g, gyro in rad/s) |
| `/labels` | (5125,) | int8 | Unified-corpus activity label (see §5) |
| `/fall_mask` | (5125,) | bool | All False — UCI HAR contains no fall events |
| `/subject_ids` | (5125,) | int16 | Globally namespaced subject IDs, 2301–2330 |
| `/trial_ids` | (5125,) | int16 | All zero — UCI HAR has no trial structure |
| `/metadata` | scalar | JSON string | Mirror of `metadata.json` |

**Channel order:** `[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]` in canonical ISB Z-up frame
(X-forward, Y-left, Z-up). Accelerometer in g (1 g = 9.81 m/s²), gyroscope in rad/s,
**gravity retained** (do not subtract).

---

## 4. Make-up

### 4.1 Subjects and Window Counts

- 30 subjects → global IDs **2301–2330** (offset +2300 per `docs/DATA.md §4.1`)
- **5,125 windows** total (after resampling and re-windowing at the corpus stride)
- 0 fall windows (UCI HAR has no falls)

### 4.2 Label Distribution (unified labels)

| Unified label | Activity | Windows |
|---|---|---|
| 1 | walking | 889 |
| 3 | standing | 960 |
| 4 | sitting | 888 |
| 5 | lying | 946 |
| 8 | stairs_up | 781 |
| 9 | stairs_down | 661 |
| **Total** | | **5,125** |

### 4.3 Channel Statistics (post-scaling, on `/windows`)

Values are unitless after fixed-range scaling. All channels well within [−1, 1].

| Channel | min | max | mean | std |
|---|---:|---:|---:|---:|
| acc_x | −0.0770 | 0.1042 | −0.00142 | 0.02486 |
| acc_y | −0.0818 | 0.1029 | −0.00536 | 0.02152 |
| acc_z | −0.0298 | 0.1364 | 0.05049 | 0.02578 |
| gyr_x | −0.1647 | 0.1712 | −0.00001 | 0.01066 |
| gyr_y | −0.0678 | 0.0794 | −0.00003 | 0.00733 |
| gyr_z | −0.1127 | 0.1018 | −0.00012 | 0.01174 |

(`acc_z` mean ≈ +0.05 reflects retained 1 g gravity component along the up-axis after
fixed scaling by ±16 g range: `1 / 16 ≈ 0.0625`.)

---

## 5. Label Mapping

UCI HAR raw activity IDs (1–6) → unified TTL-TSFM corpus labels:

| UCI HAR ID | UCI HAR activity | Unified ID | Unified name |
|---:|---|---:|---|
| 1 | WALKING | 1 | walking |
| 2 | WALKING_UPSTAIRS | 8 | stairs_up |
| 3 | WALKING_DOWNSTAIRS | 9 | stairs_down |
| 4 | SITTING | 4 | sitting |
| 5 | STANDING | 3 | standing |
| 6 | LAYING | 5 | lying |

Full corpus label map: `docs/DATA.md §2`.

---

## 6. Processing Pipeline

Implemented by `UCIHARParser(IMUDataset)` in `src/datasets/uci_har.py`. Step by step:

### 6.1 Download
- Fetch the UCI ML Repository ZIP (URL above) into `data/raw/uci_har/`.
- Extract — yields `UCI HAR Dataset/{train,test}/Inertial Signals/*.txt`.
- Idempotent: skips download if both `train` and `test` Inertial Signals folders already
  contain ≥ 9 `.txt` files.

### 6.2 Signal selection
Six raw files per split are loaded and stacked along the channel axis:

```
total_acc_{x,y,z}_<split>.txt   →  acc_{x,y,z}   (gravity-retained, in g)
body_gyro_{x,y,z}_<split>.txt   →  gyr_{x,y,z}   (in rad/s)
```

`body_acc_*` files are **intentionally ignored** — they have gravity subtracted, which would
violate the corpus convention of retaining gravity.

### 6.3 Continuous-signal reconstruction
UCI HAR ships pre-windowed at 128 samples / 64 stride (50 % overlap). The parser inverts
this overlap to recover continuous per-subject 50 Hz streams:
- For each subject, take the **first 64 samples** of every window except the last.
- For the last window, take all 128 samples to capture the tail.
- Concatenate to a single (N, 6) per-subject signal.
- Labels are broadcast across all samples in their source window (per-window label, not
  per-sample, since UCI HAR labels are window-level).

This is implemented in `_reconstruct_continuous()` (uci_har.py:96).

### 6.4 Train + test merge
UCI HAR's official 21/9 subject train/test split is **merged** — for self-supervised
pretraining we use all 30 subjects. Splitting by subject for downstream tasks happens
later in the training pipeline (see `src/training/dataset.py`).

### 6.5 Subject ID namespacing
Raw subject IDs (1–30) are shifted by **+2300** to globally unique IDs **2301–2330**
(applied inside `_reconstruct_continuous()`). Namespace table: `docs/DATA.md §4.1`.

### 6.6 Axis frame
UCI HAR's smartphone frame already aligns with the canonical ISB Z-up frame after
the smartphone is clipped to the waist in the documented orientation, so
`axis_transform_applied = "identity"` in metadata. (The class-level `AXIS_ROTATION`
constant in `uci_har.py` is informational only and is not applied.)

### 6.7 Resampling 50 Hz → 100 Hz
Performed in the base class via `scipy.signal.resample_poly(up=2, down=1)` —
polyphase resampler with built-in anti-aliasing. No additional low-pass filter.

### 6.8 Windowing
Performed in `IMUDataset.window()` after resampling:
- **Length:** 512 samples (5.12 s at 100 Hz)
- **Stride:** 256 samples (50 % overlap) — pretraining stride
- **Fall stride:** 50 samples — unused here (no fall events)
- **Fall mask rule:** any-overlap (any sample with a fall label flips the bit) — all False here
- Output dtype `float32`, shape `(N, 512, 6)`

### 6.9 Fixed-range scaling
After windowing, raw physical-unit windows are duplicated to `/windows_raw` and the
scaled view is stored in `/windows`:

```python
ACC_RANGE_G  = 16.0       # ±16 g full scale
GYRO_RANGE   = 34.906585  # ±2000 dps in rad/s
windows[..., :3] /= ACC_RANGE_G   # → [−1, 1]
windows[..., 3:] /= GYRO_RANGE    # → [−1, 1]
```

This is **not** per-window min-max — the same divisor is used corpus-wide so amplitudes
remain comparable across datasets and subjects. (Inference-time `use_norm=True` adds a
deprecated per-instance min-max on top; see `docs/DATA.md §1` and
`docs/prod_model_card.md §6.2`.)

### 6.10 HDF5 write
Written by `IMUDataset._write_hdf5()` with chunks `(256, 512, 6)`, LZF compression, and
`metadata.json` mirrored into the `/metadata` attribute as a JSON string.

---

## 7. Validation

Validation passed all 22 checks on 2026-04-11 (`stats.json`):
- HDF5 readable, all required keys present with correct shapes/dtypes.
- `/windows` in [−1, 1] (actual [−0.165, 0.171]); no NaN/Inf in either windowed array.
- All labels resolve to unified-map keys (1, 3, 4, 5, 8, 9).
- `fall_mask.sum() == 0` matches metadata `n_fall_windows: 0`.
- 5125 windows match the array shape.
- Subject ID range 2301–2330 (30 unique).

Re-run via the `dataset-validator` agent or `scripts/checks/validate_corpus.py`.

---

## 8. Reproduction

```bash
uv run python -m src.datasets.uci_har
```

Idempotent: skips download if the raw archive is already extracted, then rewrites
`windows.h5`, `metadata.json` from scratch. Total runtime ≈ 1 minute on the GCP A100 box.

---

## 9. Caveats

- **Activity labels are window-level**, not sample-level — UCI HAR collected discrete
  activity sessions, so the per-sample labels emitted by the reconstruction step are
  broadcast from the source window's label rather than from a continuous annotation.
- **No fall events.** UCI HAR is an ADL-only benchmark; `fall_mask` is identically False.
  Used in the corpus for activity diversity, not fall supervision.
- **Smartphone, not research-grade IMU.** Lower bandwidth and noisier than the
  research IMUs in WearGait-PD / Mobilise-D / Voisard.
- **Single trial structure dropped.** UCI HAR's two-trials-per-subject protocol is not
  preserved (`trial_ids` all zero) — the original split is collapsed during continuous-
  signal reconstruction.
- **Pre-windowed source.** The 50 % overlap inherent in the UCI HAR distribution means
  reconstructed continuous signals contain stitch boundaries every 64 samples
  (1.28 s at 50 Hz). For pretraining at 5.12 s windows this is far below window length
  and not expected to bias features.
