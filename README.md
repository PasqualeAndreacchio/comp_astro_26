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

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> --transit [--output lightcurve.png]
```

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-t, --transit`: Generate a transit light curve from the YAML input
- `-d, --detection [METHOD]`: Run exoplanet detection using specified classifier (`rf` for Random Forest, `cnn` for Convolutional Neural Network; default: `rf`)
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for transit plot (default: `<slug>_light_curve.png`) or evaluation report for detection

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
