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
- `-d, --detect`: Reserved for the course detection exercise
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for the transit plot (default: `lc.png`)

### Examples

```bash
# Generate transit light curve for HD 209458
daneel -i examples/HD_209458_b.yaml -t

# Customise output filename
daneel -i examples/HD_209458_b.yaml -t -o HD_209458_b_lightcurve.png

# Run transit light curve with default examples parameters
daneel -i examples/params.yaml --transit --output lightcurve.png

# Run atmospheric characterization
daneel -i examples/params.yaml --detect

# Run atmospheric analysis
daneel -i examples/params.yaml --atmosphere
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

# Customise static PNG suffix (e.g. produces WASP_12_b_taskG.png)
python src/daneel/detection/transit.py "WASP-12 b" --suffix taskG
```

Command-line options for `transit.py`:
- `planet`: Target exoplanet name (e.g. `'TRAPPIST-1 b'`, `'WASP-12 b'`; default: `'HD 209458 b'`)
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
