# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling exoplanet transits and building detection
pipelines. The repository contains the course example code, a runnable CLI,
and Sphinx documentation.

Documentation: <https://tiziano1590.github.io/comp_astro_26/>.

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_26.git
cd comp_astro_26
pip install .
```

### Development installation

```bash
pip install -e '.[dev,docs]'
```

## Project Structure

The repository is organized into modular packages separating the command-line interface, analytical transit physics, machine learning detection pipelines, parameter parsing, and shared utilities:

```text
comp_astro_26/
├── src/
│   └── daneel/
│       ├── __init__.py           # Package initialization and top-level exports
│       ├── __main__.py           # CLI entry point (daneel -i ... -t / -d / --dream)
│       ├── detection/            # Transit modeling & ML detection subpackage
│       │   ├── __init__.py       # Exports detectors and modeling functions
│       │   ├── base.py           # BaseTransitDetector (abstract lifecycle & evaluation)
│       │   ├── random_forest.py  # RandomForestDetection (scikit-learn pipeline)
│       │   ├── cnn.py            # CNNDetection (1D PyTorch ConvNet pipeline)
│       │   ├── data_process.py   # Dataset loading, stratified splitting & augmentation
│       │   ├── transit.py        # NASA Exoplanet Archive queries & transit animations
│       │   ├── transit_model.py  # BATMAN-based transit analytical light curves
│       │   └── classifiers.py    # Detection interfaces and aliases
│       ├── dream.py              # DreamGAN (synthetic transit light curve generation via DCGAN)
│       ├── parameters/           # Configuration handling
│       │   ├── __init__.py
│       │   └── parameters.py     # Parameters class for YAML loading & sanitization
│       └── utils/                # Shared cross-cutting utilities
│           ├── __init__.py
│           └── utils.py          # Logging setup, deep_merge, and config normalization
├── docs/                         # Sphinx documentation source and build configuration
├── examples/                     # Course example YAML configurations and targets
├── tests/                        # Test suite (pytest)
├── pyproject.toml                # Package metadata, entry points, and dependencies
└── README.md                     # Project overview and documentation
```

### Module Overview & Architecture

- **`daneel.detection.base` (`BaseTransitDetector`)**: An abstract base class (`ABC`) defining the standard end-to-end detection lifecycle (`load_data` → `build_model` → `fit` → `evaluate` → `display_statistics` → `save_report`). It encapsulates optimal decision threshold tuning via Youden's J statistic, ROC-AUC, classification metrics, confusion matrices, and file exports, while exposing extensible lifecycle hooks for subclass customization.
- **`daneel.detection.random_forest` (`RandomForestDetection`)**: Inherits from `BaseTransitDetector`. Builds a scikit-learn `Pipeline` (with optional `StandardScaler` and `RandomForestClassifier`), fits the ensemble on light curve features, predicts probabilities, and hooks Out-of-Bag (OOB) generalization scores into statistics and reports.
- **`daneel.detection.cnn` (`CNNDetection`)**: Inherits from `BaseTransitDetector`. Builds a 1D Convolutional Neural Network (`_Conv1DBlock` backbone with pooling, followed by a dropout-regularized fully-connected head), manages GPU/CPU training with cosine annealing and positive-class weighting, and hooks epoch loss/accuracy curves into reporting.
- **`daneel.detection.data_process`**: Centralized data management module providing `load_data` (stratified train/test splits, light curve binning), `create_balanced_dataset` (photometric data augmentation via phase roll, amplitude scaling, and Gaussian jitter), and `resolve_file_path`.
- **`daneel.detection.transit` & `transit_model`**: Analytical transit modeling routines. `transit.py` queries the NASA Exoplanet Archive TAP API to fetch physical parameters, generates transit light curves, and renders synchronized sky-plane transit animations. `transit_model.py` provides `batman`-based analytical modeling.
- **`daneel.utils.utils`**: Shared helper functions supporting the pipeline: `setup_logging` (uniform application-wide logging), `deep_merge` (recursive dictionary merging), and `normalize_config` (flexible YAML normalization supporting nested, flat, or wrapped schemas).
- **`daneel.dream` (`DreamGAN`)**: Generative Adversarial Network pipeline adapting DCGAN architectures to synthesize photometric transit light curves. It handles preprocessing (truncation of 2000-sample sequences to 1936 points and reshaping into $44\times 44$ images), robust dynamic range scaling, DCGAN training with one-sided label smoothing, 1D curve reconstruction, and publication-ready multi-panel visualization.
- **`daneel.parameters.parameters`**: Provides the `Parameters` class for YAML file loading, type validation, and sanitization.

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> --transit [--output lightcurve.png]
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-t, --transit`: Generate a transit light curve from the YAML input
- `-d, --detection [METHOD]`: Run exoplanet detection using specified classifier (`rf` for Random Forest, `cnn` for Convolutional Neural Network; default: `rf`)
- `--dream`: Train the GAN and generate synthetic transit light curves using the YAML configuration
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for transit plot (default: `<slug>_light_curve.png`), evaluation report for detection, or synthetic transit plot for dream (default: `dream_synthetic_transits.png`)

### Examples

```bash
# Generate transit light curve for HD 209458
daneel -i examples/HD_209458_b.yaml -t

# Customise output filename
daneel -i examples/HD_209458_b.yaml -t -o HD_209458_b_lightcurve.png

# Run transit light curve with default examples parameters
daneel -i examples/params.yaml --transit --output lightcurve.png

# Run Random Forest exoplanet detection
daneel -i random_forest/parameters/config1_shallow_trees.yaml -d rf

# Equivalent syntax with full flag name
daneel -i random_forest/parameters/config1_shallow_trees.yaml --detection rf

# Customise output evaluation report path
daneel -i random_forest/parameters/config1_shallow_trees.yaml -d rf -o custom_report.txt

# Run CNN exoplanet detection
daneel -i cnn/parameters/config_default.yaml -d cnn

# Equivalent syntax with full flag name
daneel -i cnn/parameters/config_default.yaml --detection cnn

# Customise output evaluation report path for CNN
daneel -i cnn/parameters/config_default.yaml -d cnn -o custom_cnn_report.txt

# Train GAN and generate synthetic transit light curves (DreamGAN)
daneel -i examples/params.yaml --dream

# Run DreamGAN with custom output plot path
daneel -i examples/dream.yaml --dream -o custom_synthetic_transits.png

# Run atmospheric analysis
daneel -i examples/params.yaml --atmosphere
```

### Exoplanet Transit Detection (Random Forest)

Daneel includes an end-to-end Machine Learning classification pipeline inside `daneel.detection` designed for binary transit detection (Planet vs Non-Planet) using TESS photometric light curves.

#### Features
- **Configurable YAML execution**: Model hyperparameters, data splitting, and output destinations can be fully defined in YAML.
- **Photometric Data Augmentation**: Automatically balances the training set via Gaussian jitter, amplitude scaling, and transit phase roll to address severe astrophysical class imbalance.
- **Pipeline Standardization**: scikit-learn `Pipeline` preventing data leakage with optional `StandardScaler`.
- **Optimal Threshold Tuning**: Computes the optimal decision threshold using Youden's J statistic ($J = \text{TPR} - \text{FPR}$) alongside default 0.5 threshold evaluation.
- **Comprehensive Logging & Reports**: Emits structured `INFO` logging, confusion matrix, precision metrics per class, and saves a clean evaluation report.

#### CLI Usage

```bash
# Run Random Forest detection from a YAML parameter file
daneel -i random_forest/parameters/config1_shallow_trees.yaml -d rf

# Long-form flag
daneel -i random_forest/parameters/config1_shallow_trees.yaml --detection rf
```

#### Python API Usage

```python
from daneel.detection import RandomForestDetection

# Initialise from YAML configuration file and run end-to-end
detector = RandomForestDetection(params="random_forest/parameters/config1_shallow_trees.yaml")
detector.run(output_file="results/evaluation_report.txt")

# Or run modularly step-by-step
detector = RandomForestDetection(
    csv_path="tess_data.csv",
    n_estimators=300,
    max_depth=15,
    standardize=False,
)
detector.load_data()
detector.fit()
metrics = detector.evaluate()
detector.display_statistics()
detector.save_report("report.txt")
```

#### YAML Parameter Structure

```yaml
data:
  csv_path: "tess_data.csv"           # Path to input light curve dataset
  n_bins: 1000                        # Flux feature bins (flux_0000..flux_0999)
  test_size: 0.2                      # Stratified test set ratio
  balanced: true                      # Enable data augmentation balancing
  samples_per_class: 350              # Target samples per class in training set
  random_state: 42                    # Deterministic split seed

model:
  standardize: false                  # Whether to include StandardScaler
  n_estimators: 100                   # Number of decision trees
  max_depth: 10                       # Maximum tree depth
  min_samples_leaf: 1                 # Minimum samples per leaf node
  max_features: "sqrt"                # Number of features per split
  bootstrap: true                     # Bootstrap sampling
  class_weight: null                  # Class weighting scheme
  random_state: 42                    # Ensemble random seed
  n_jobs: -1                          # Parallel CPU workers
  oob_score: true                     # Compute Out-of-Bag score

output:
  report_path: "random_forest/results/report_config1_shallow_trees.txt"

logging:
  level: "INFO"
```

### Exoplanet Transit Detection (Convolutional Neural Network — CNN)

Daneel includes an end-to-end Deep Learning 1D-CNN pipeline inside `daneel.detection` designed for binary transit detection (Planet vs Non-Planet) using folded, phase-binned photometric light curves.

#### Features
- **Configurable YAML execution**: Data splitting, augmentation, convolutional network backbone, fully-connected head, and training hyperparameters can be fully defined in YAML.
- **Photometric Data Augmentation**: Automatically balances the training set via Gaussian jitter, amplitude scaling, and transit phase roll to address severe astrophysical class imbalance.
- **Deep 1D Convolutional Architecture**: Configurable stack of `Conv1d` → `BatchNorm1d` → `ReLU` → `MaxPool1d` feature extraction blocks followed by fully-connected layers with Dropout and `BCEWithLogitsLoss`.
- **Optimal Threshold Tuning**: Computes the optimal decision threshold using Youden's J statistic ($J = \text{TPR} - \text{FPR}$) alongside default 0.5 threshold evaluation.
- **Comprehensive Logging & Reports**: Emits structured `INFO` logging (with training progress printed every 10 epochs), confusion matrix, precision metrics per class, and saves a clean evaluation report.

#### CLI Usage

```bash
# Run CNN detection from a YAML parameter file
daneel -i cnn/parameters/config_default.yaml -d cnn

# Long-form flag
daneel -i cnn/parameters/config_default.yaml --detection cnn

# Customise output evaluation report path
daneel -i cnn/parameters/config_default.yaml -d cnn -o cnn/results/report_default.txt
```

#### Python API Usage

```python
from daneel.detection import CNNDetection

# Initialise from YAML configuration file and run end-to-end
detector = CNNDetection(params="cnn/parameters/config_default.yaml")
detector.run(output_file="results/cnn_evaluation_report.txt")

# Or run modularly step-by-step
detector = CNNDetection(
    csv_path="tess_data.csv",
    conv_channels=[16, 32, 64],
    kernel_size=7,
    pool_size=2,
    fc_units=[128, 64],
    dropout=0.3,
    batch_size=32,
    epochs=30,
    learning_rate=0.001,
)
detector.load_data()
detector.build_model()
detector.fit()
metrics = detector.evaluate()
detector.display_statistics()
detector.save_report("report.txt")
```

#### YAML Parameter Structure

```yaml
data:
  csv_path: "tess_data.csv"           # Path to input light curve dataset
  n_bins: 1000                        # Flux feature bins (flux_0000..flux_0999)
  test_size: 0.2                      # Stratified test set ratio
  balanced: true                      # Enable data augmentation balancing
  samples_per_class: 350              # Target samples per class in training set
  random_state: 42                    # Deterministic split seed

model:
  conv_channels: [16, 32, 64]         # Out-channels for each Conv1D block
  kernel_size: 7                      # Convolution kernel width
  pool_size: 2                        # MaxPool1d window size
  fc_units: [128, 64]                 # Fully-connected layer widths
  dropout: 0.3                        # Dropout probability
  batch_size: 32                      # Training batch size
  epochs: 30                          # Total training epochs
  learning_rate: 0.001                # Adam initial learning rate
  weight_decay: 0.0001                # L2 regularisation strength
  random_state: 42                    # Deterministic seed for PyTorch and NumPy

output:
  report_path: "cnn/results/report_default.txt"   # Output evaluation report path
  model_path: null                                 # Optional path to save model weights (.pt)

logging:
  level: "INFO"
```

### Synthetic Transit Light Curve Generation (DreamGAN)

Daneel includes a deep generative pipeline in `daneel.dream` (`DreamGAN`) designed to synthesize realistic exoplanet transit light curves using a Deep Convolutional Generative Adversarial Network (DCGAN). The module preprocesses 1D photometric signals into 2D square representations ($44 \times 44$), trains a custom DCGAN, and reconstructs synthetic curves back into continuous 1D light curves with physical flux scaling.

#### Features

- **Configurable YAML Execution**: Training data paths, dataset filtering, GAN architectural channels (`ngf`, `ndf`, `nz`), training hyperparameters (`epochs`, `batch_size`, `lr`, `beta1`), and output destinations are fully configurable via YAML.
- **Square-Input Preprocessing & Reshaping**: Ingests 2000-sample photometric curves (`flux_*`), filters confirmed transits (`label=1`), truncates them to 1936 samples ($44 \times 44$), and rescales data robustly to $[-1, 1]$ matching the generator's `Tanh` activation function.
- **DCGAN Architecture**: Custom Generator combining linear projection with progressive transposed convolutions (`ConvTranspose2d`: $11\times 11 \to 22\times 22 \to 44\times 44$), Batch Normalization, and ReLU activations; coupled with a Discriminator applying strided convolutions (`Conv2d`), LeakyReLU(0.2), spatial Dropout, and Adaptive Average Pooling.
- **Small-Dataset Regularization**: Implements one-sided real label smoothing (targeting $0.9$ instead of $1.0$), weight decay ($10^{-4}$ on the discriminator), and Adam momentum tuning ($\beta_1 = 0.5$) to stabilize adversarial training on small astrophysical sample sizes ($N = 472$).
- **1D Reconstruction & Inverse Scaling**: Decodes synthetic $44\times 44$ matrices back into 1936-sample vectors and applies exact inverse denormalization to recover physical relative flux units.
- **Multi-Panel Diagnostic Plotting**: Automatically generates a 4-panel diagnostic figure displaying 2D synthetic representations, 1D reconstructed light curves with ensemble medians, real TESS baseline curves, and loss progression.

#### CLI Usage

```bash
# Run DreamGAN from the default parameters file
daneel -i examples/params.yaml --dream

# Run DreamGAN using a standalone dream configuration
daneel -i examples/dream.yaml --dream

# Customise output synthetic plot path
daneel -i examples/dream.yaml --dream -o results/synthetic_transits.png
```

#### Python API Usage

```python
from daneel.dream import DreamGAN

# Initialise from YAML configuration file and run end-to-end
dreamer = DreamGAN(params="examples/dream.yaml")
dreamer.run(output_file="dream_synthetic_transits.png")

# Or run modularly step-by-step
dreamer = DreamGAN(
    data_path="tess_data.csv",
    filter_transits=True,
    epochs=30,
    batch_size=32,
    lr=0.0002,
    beta1=0.5,
    nz=100,
    ngf=32,
    ndf=32,
)
dataset = dreamer.load_data()
dreamer.train()
synth_2d, synth_1d = dreamer.generate(num_samples=16)
dreamer.plot_results("dream_synthetic_transits.png")
```

#### YAML Parameter Structure

The `dream` block can be included in `examples/params.yaml` or as a standalone file (e.g., `examples/dream.yaml`):

```yaml
dream:
  data_path: "tess_data.csv"           # Path to TESS light-curve CSV dataset
  filter_transits: true                # Filter for confirmed transit light curves (label=1)
  target_length: 1936                  # 44 x 44 square input length
  side_length: 44                      # Grid side dimension
  epochs: 30                           # Training epochs
  batch_size: 32                       # Batch size for gradient updates
  lr: 0.0002                           # Initial learning rate for Adam optimiser
  beta1: 0.5                           # Beta1 momentum hyperparameter for Adam
  nz: 100                              # Latent vector z dimensionality
  ngf: 32                              # Generator feature map channel depth
  ndf: 32                              # Discriminator feature map channel depth
  dropout: 0.25                        # Dropout rate applied in discriminator blocks
  random_state: 42                     # Deterministic random seed
  num_generated: 16                    # Number of synthetic curves to plot
  output_plot: "dream_synthetic_transits.png" # Output figure filepath
```

#### Outputs

When executing `--dream`, Daneel saves a 4-panel diagnostic figure (default: `dream_synthetic_transits.png`):
1. **Synthetic 2D Light Curves ($44\times 44$)**: Grid of 16 generated 2D square light curve representations produced by the Generator.
2. **Reconstructed 1D Synthetic Light Curves**: Individual reconstructed 1D light curves (1936 points) denormalized back to relative flux, overlaid with the ensemble synthetic median profile.
3. **Real Transit Light Curves**: Sample of real TESS confirmed transit curves ($N = 472$) and their median profile for visual fidelity comparison.
4. **GAN Training Loss Curves**: Generator and Discriminator Binary Cross-Entropy loss curves tracked over all training epochs.

---

### Technical Limitations of the DreamGAN Approach

While folding 1D time series into 2D grids enables the direct reuse of established 2D Computer Vision architectures (such as DCGAN), this formulation introduces structural and physical trade-offs:

#### 1. Limitations of the $44 \times 44$ Square Representation for 1D Light Curves

- **Data Truncation**: Truncating 2000-point observation sequences to $1936 = 44^2$ samples permanently discards 64 data points ($3.2\%$ of the curve). If transits occur near the sequence boundaries or require extensive out-of-transit baseline for normalisation, this truncation risks shaving off transit wings or critical calibration baseline.
- **Artificial 2D Spatial Locality & Broken Continuity**: Wrapping a 1D time series row-by-row into a $44\times 44$ matrix breaks continuous time. Two non-adjacent temporal samples separated by 44 time steps ($t$ and $t+44$) are placed vertically adjacent, leading 2D convolutional kernels ($3\times 3$ or $4\times 4$) to treat them as local spatial neighbors. Conversely, truly consecutive points at row boundaries (index $44k - 1$ and $44k$) are split across opposite ends of the matrix, severing temporal adjacency.
- **Loss of 1D Temporal Shift-Invariance**: 2D convolutions assume spatial translation invariance along both horizontal and vertical axes. In an unfolded light curve, vertical shifts represent discontinuous jumps of 44 time steps rather than spatial translation. As a result, 2D filters cannot distinguish continuous transit dips from folded boundary artifacts.
- **Checkerboard Artifacts**: Transposed convolutions (`ConvTranspose2d`) frequently produce high-frequency checkerboard patterns in 2D space. When unrolled back to 1D, these 2D artifacts manifest as periodic high-frequency noise spikes across the reconstructed light curve.

#### 2. Limitations of the GAN & Dataset

- **Small Sample Size ($N = 472$ Transits)**: Deep Convolutional GANs typically require tens of thousands of images (e.g., CelebA $>200{,}000$) to generalize well. Training on only 472 un-phased transit curves creates a high risk of discriminator memorization, mode collapse (producing near-identical curves), or elevated high-frequency noise.
- **Training Instability**: Adversarial min-max optimization is inherently sensitive to hyperparameters (learning rates, batch size, normalization bounds). Small perturbations can cause the discriminator to overpower the generator early in training, stalling gradient updates.
- **Absence of Physical Constraints**: The GAN optimizes purely statistical pixel distributions without enforcing physical laws of planetary transits (e.g., Keplerian orbital geometry, stellar limb darkening models, ingress/egress symmetry, or conservation of flux). Consequently, synthetic curves may exhibit non-astrophysical distortions unless constrained by physical analytical models (such as `batman`).

### Automated Transit Modelling & NASA Archive Integration

You can model transits for any confirmed exoplanet automatically by passing its name. The script queries the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) (TAP API `pscomppars` table) to fetch physical and orbital parameters ($P$, $R_p/R_*$, $a/R_*$, $i$, $e$, $\omega$, and transit duration $T_{\rm dur}$), dynamically scales the observation time window, and generates both a static light curve and a synchronized sky-plane transit animation GIF:

```bash
# Default target (HD 209458 b)
python src/daneel/detection/transit.py

# Query archive for any confirmed exoplanet
python src/daneel/detection/transit.py "TRAPPIST-1 b"
python src/daneel/detection/transit.py "Kepler-7 b"
python src/daneel/detection/transit.py "WASP-12 b"

# Model transit using a YAML parameter file
python src/daneel/detection/transit.py -c examples/wasp43b_params.yaml

# Customise static PNG suffix (e.g. produces WASP_12_b_taskG.png)
python src/daneel/detection/transit.py "WASP-12 b" --suffix taskG
```

Command-line options for `transit.py`:
- `planet`: Target exoplanet name (e.g. `'TRAPPIST-1 b'`, `'WASP-12 b'`; default: `'HD 209458 b'` or inferred from `--config`)
- `-i, -c, --input, --config`: Path to YAML parameter file containing transit parameters
- `--suffix`: Suffix for the generated static light curve PNG (default: `'light_curve'`)
- `--local`: Force using local default parameters instead of querying the online archive

Outputs generated:
- `<slug>_<suffix>.png`: High-resolution light curve with physical parameters and ingress/egress markers.
- `<slug>_transit.gif`: Synchronized 2D sky-plane animation of the planet traversing the stellar disk alongside the real-time light curve progression.

You can also call it directly from Python:

```python
from daneel.detection.transit import run_for_planet

# Query NASA Archive and generate plot + GIF
run_for_planet("TRAPPIST-1 b")
```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

## Development

Run the test suite and style checks with:

```bash
pytest
ruff check .
```

Build the documentation with `make -C docs html`.

## License

This project is licensed under the GNU General Public License v3.0.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)
