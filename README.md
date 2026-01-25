# Data Quality POC

A proof-of-concept demonstrating data quality detection capabilities using real-world open datasets corrupted with realistic production data issues.

## Purpose

This POC demonstrates:
1. Real-world data corrupted with common production issues
2. Blind detection of data quality problems from raw data alone
3. Range of dataset sizes from 10K to 900K rows

## Important: Testing with Agentic Coding Tools

**Make sure to turn off web tool use when testing with agentic coding assistants.** The corruption details are intentionally hosted externally so that the AI cannot "cheat" by reading this README to discover what issues exist in each dataset.

## Datasets

### Summary

| Dataset | Rows | Size | Domain | Corruption Theme |
|---------|------|------|--------|------------------|
| `nyc_taxi_v1.csv` | 900,000 | 90MB | Transportation | Timestamps, impossible values |
| `online_retail_v1.csv` | 541,909 | 45MB | E-commerce | Encoding, text issues |
| `chicago_crimes_v1.csv` | 393,406 | 95MB | Public Safety | Geographic corruption |
| `air_quality_v1.csv` | 250,000 | 78MB | Environmental | Sensor/unit issues |
| `chicago_crimes_v2.csv` | 103,000 | 24MB | Public Safety | Temporal, categorical |
| `nyc_taxi_v2.csv` | 54,600 | 5.5MB | Transportation | Duplicates, nulls |
| `online_retail_v2.csv` | 25,000 | 2.3MB | E-commerce | Business logic |
| `air_quality_v2.csv` | 10,920 | 3.4MB | Environmental | Temporal, geographic |

### Data Sources

| Dataset | Source | License |
|---------|--------|---------|
| NYC Yellow Taxi | [NYC TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Public Domain |
| Online Retail | [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) | CC BY 4.0 |
| Chicago Crimes | [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) | Public Domain |
| EPA Air Quality | [EPA AQS](https://aqs.epa.gov/aqsweb/airdata/download_files.html) | Public Domain |

---

## Corruption Details

For the detailed corruption manifest (what issues were injected and at what percentages), see:

**[Corruption Details Gist](https://gist.github.com/Hermetian/df1c8d1172c9c9340fac1c8d676ce2f2)**

---

## Usage

### Testing Data Quality Detection

Download any corrupted CSV and pass it to a fresh analysis session to identify issues without prior knowledge of the corruption.

### Regenerating Data

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas pyarrow openpyxl requests
python3 corrupt_datasets.py
```

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
