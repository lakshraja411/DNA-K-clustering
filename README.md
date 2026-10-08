# DNA K-Means Workbench v1.0

A standalone Streamlit app for one nanopore recording at a time.

## Workflow

1. Upload:
   - `*.dataset.npz`
   - `*.event_data.npz`
   - `*.event_fitting.npz`
2. Match files by event start timestamps.
3. Cluster using the exact NanoSense scalar descriptors stored in `dataset.npz X[:,0:7]`:
   - height
   - FWHM
   - height at FWHM
   - area
   - width
   - skew
   - kurtosis
4. Scale each feature to `[-1, 1]`.
5. Run reproducible NumPy K-means with either:
   - manual `k`
   - silhouette-selected `k`
6. Inspect physical plots, PCA/loadings and real midpoint-aligned waveform families.
7. Create ΔI/current gates from the original undivided dataset and inspect dwell-time histograms.
8. Calculate segment-weighted event metrics from:
   - `SEGMENT_INFO_*_segment_mean_diffs`
   - `SEGMENT_INFO_*_segment_widths_time`
9. Export K-means clusters and ΔI gates as three NPZ categories plus CSV/provenance.

## Segment-weighted definitions

For event segments `j` with blockade `ΔI_j` and duration `τ_j`:

### Duration-weighted mean blockade
`Σ(ΔI_j τ_j) / Στ_j`

This is the primary weighted current quantity.

### Total segmented dwell
`Στ_j`

This is the recommended primary time quantity.

### Duration-weighted mean segment duration
`Στ_j² / Στ_j`

This deliberately gives more weight to long-lived segments.

### Blockade-weighted mean segment duration
`Σ(|ΔI_j| τ_j) / Σ|ΔI_j|`

This is exploratory: deeper-blockade segments receive greater weight.

## GitHub / Streamlit deployment

Put these files in one GitHub repository:

- `app.py`
- `core.py`
- `requirements.txt`

Then on Streamlit Community Cloud:
1. New app
2. Select the repository
3. Main file: `app.py`
4. Deploy

## Important interpretation

The clustering uses the NanoSense scalar descriptors exactly as stored in the dataset file. It does **not** claim to duplicate the hidden waveform clustering implementation of the separate NanoSense clustering GUI.

Exported subsets preserve original dataset rows and event-indexed arrays where matching succeeds. Original event IDs remain unchanged.


## New in this version

- Midpoint-aligned **waveform family panel plots** for each cluster:
  - individual family panels with real member traces in grey and the median representative in red
  - combined overlay plot
  - x-axis can be either time relative to the event midpoint or centered data index
- Added **dwell-time histograms by cluster**
- Added **blockade histograms by cluster**
- Added a **multi-salt comparison** page:
  - upload many `.npz` files at once
  - the app groups matching `dataset`, `event_data`, and `event_fitting` files by filename stem
  - runs the same K-means settings on each salt
  - compares family-resolved blockade and dwell across salts
  - compares matched median family shapes across salts


## Multi-salt upload fix

The multi-salt page now uses explicit three-file upload slots for LiCl, NaCl, KCl, RbCl and CsCl.
It no longer assumes that dataset, event-data and event-fitting filenames have identical timestamps.

The multi-salt page now shows:
- full individual plots for every loaded salt;
- cluster waveform family panels and overlay;
- blockade-vs-dwell plots;
- dwell and blockade histograms;
- population fractions;
- explicit user-confirmed mapping from salt-specific clusters to common Family A/B/C... labels;
- combined shared-axis blockade–dwell panels;
- matched waveform-family profiles across salts;
- family population fractions;
- family-resolved dwell with event-level IQR;
- family-resolved blockade with event-level IQR;
- dwell and blockade heatmaps.


## v3 additions

- Cluster and salt summary tables now report both **mean and median** for dwell, blockade, FWHM and area where relevant.
- Dwell/blockade histograms show both:
  - solid vertical line = mean
  - dashed vertical line = median
- Waveform family plots can show **Mean**, **Median**, or **Both** representative profiles.
- Cross-salt family waveform comparison can use either mean or median representative profiles.
- Cross-salt family dwell/blockade plots can display:
  - Median + IQR
  - Mean ± SD
  - Mean ± SEM
- Segment-weighted analysis is now distribution-first:
  - individual-salt weighted ΔI histograms
  - individual-salt total segmented dwell histograms
  - combined all-salt overlaid histograms
  - all-salt error-bar summaries using median+IQR, mean±SD, or mean±SEM
  - all-salt weighted ΔI versus segmented dwell scatter
- Reusable **Edit plot appearance** controls were added throughout the main analysis:
  - plot title
  - x-axis title
  - y-axis title
  - x min / x max
  - y min / y max
  - histogram bin count where relevant
  - optional log axes on applicable scatter plots
