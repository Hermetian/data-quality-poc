# Data Quality POC

A proof-of-concept demonstrating data quality detection capabilities using real-world open datasets corrupted with realistic production data issues.

## Purpose

This POC demonstrates:
1. Real-world data corrupted with common production issues
2. Blind detection of data quality problems from raw data alone
3. Range of dataset sizes from 10K to 900K rows

## Important: Testing with Agentic Coding Tools

**Make sure to turn off web tool use when testing with agentic coding assistants.** The corruption details are intentionally hosted externally so that the AI cannot "cheat" by reading this README to discover what issues exist in each dataset.

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install pandas pyarrow openpyxl requests

# Download raw datasets from public sources
python download_datasets.py

# Generate corrupted datasets (standard mode)
python corrupt_datasets.py

# Or with options (see below)
python corrupt_datasets.py --randomize --control --include-rare
```

## Corruption Options

The corruption script supports several modes:

| Option | Description |
|--------|-------------|
| `--randomize` | Randomly select 1-8 corruptions per dataset instead of all |
| `--scale FLOAT` | Scale corruption percentages (0.5 = half rate, 2.0 = double) |
| `--control` | Generate clean "control" versions alongside corrupted files |
| `--include-rare` | Include rare corruptions (0.5-1% rate) |
| `--include-timeboxed` | Include timeboxed corruptions (contiguous blocks) |
| `--seed INT` | Random seed for reproducibility (default: 42) |
| `--datasets NAME...` | Process only specific datasets |

### Example Configurations

```bash
# Standard: all corruptions at default percentages
python corrupt_datasets.py

# Randomized subset with control files for comparison
python corrupt_datasets.py --randomize --control

# Lighter corruption for easier detection tasks
python corrupt_datasets.py --scale 0.5

# Full chaos mode - everything enabled
python corrupt_datasets.py --randomize --include-rare --include-timeboxed --control

# Process only specific datasets
python corrupt_datasets.py --datasets air_quality_v1 nyc_taxi_v2
```

### Corruption Types

- **Standard**: Common issues at typical rates (3-25%)
- **Rare**: Uncommon but realistic issues (0.5-1.5%)
- **Timeboxed**: Contiguous blocks of errors simulating system outages, bad batches, etc.

## Datasets

### Summary

| Dataset | Rows | Size | Domain |
|---------|------|------|--------|
| `usgs_earthquakes_v1.csv` | 50,000 | ~9MB | Geological |
| `nyc_taxi_v1.csv` | 900,000 | 90MB | Transportation |
| `online_retail_v1.csv` | 541,909 | 45MB | E-commerce |
| `chicago_crimes_v1.csv` | 393,406 | 95MB | Public Safety |
| `air_quality_v1.csv` | 250,000 | 78MB | Environmental |
| `chicago_crimes_v2.csv` | 103,000 | 24MB | Public Safety |
| `nyc_taxi_v2.csv` | 54,600 | 5.5MB | Transportation |
| `online_retail_v2.csv` | 25,000 | 2.3MB | E-commerce |
| `air_quality_v2.csv` | 10,920 | 3.4MB | Environmental |

### Data Sources

| Dataset | Source | License |
|---------|--------|---------|
| USGS Earthquakes | [USGS Earthquake Catalog](https://earthquake.usgs.gov/fdsnws/event/1/) | Public Domain |
| NYC Yellow Taxi | [NYC TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Public Domain |
| Online Retail | [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) | CC BY 4.0 |
| Chicago Crimes | [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) | Public Domain |
| EPA Air Quality | [EPA AQS](https://aqs.epa.gov/aqsweb/airdata/download_files.html) | Public Domain |

---

## Corruption Details

For the detailed corruption manifest (what issues were injected and at what percentages), see:

**https://rentry.co/bnh3gs9m**

---

## Issue Categories

- **Timestamp/Date**: Format mixing, future dates, temporal logic violations
- **Numeric**: Negative impossibles, outliers, precision artifacts
- **Duplicates**: Exact, near-duplicates with conflicts
- **Missing/Null**: Various representations (`NULL`, `N/A`, `''`, etc.)
- **Encoding**: Special chars, whitespace, case inconsistency
- **Geographic**: Swapped coords, out-of-bounds, precision loss
- **Business Logic**: Constraint violations, impossible combinations
- **Referential Integrity**: Invalid IDs, code/description mismatches
- **Categorical**: Format variations, encoding inconsistencies

## Testing Methodology

1. **Blind Test**: Pass a corrupted CSV to a fresh analysis session (web tools disabled)
2. **Control Group**: Use `--control` to generate clean versions for comparison
3. **Difficulty Scaling**: Use `--scale` to adjust corruption intensity
4. **Randomization**: Use `--randomize` for unpredictable corruption combinations
