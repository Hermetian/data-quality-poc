# Data Quality POC

A proof-of-concept demonstrating data quality detection capabilities using real-world open datasets corrupted with realistic production data issues.

## Purpose

This POC demonstrates:
1. Real-world data that has been corrupted with common production issues
2. Blind detection of data quality problems from raw data alone

## Datasets

All datasets are sourced from real open data sources, then corrupted with realistic issues.

### Sources

| Dataset | Source | Original Size | Sample Size |
|---------|--------|---------------|-------------|
| NYC Yellow Taxi | [NYC TLC Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | 2.9M rows | 100K rows |
| Online Retail | [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail) | 541K rows | 150K rows |
| Chicago Crimes | [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) | 8M+ rows | 100K rows |
| EPA Air Quality | [EPA AQS](https://aqs.epa.gov/aqsweb/airdata/download_files.html) | 847K rows | 100K rows |

### Corrupted Versions

Each dataset has 2 corrupted versions with different issue types:

**NYC Taxi (Transportation)**
- `nyc_taxi_v1_timestamps_values.csv`: Mixed timestamp formats, future dates, negative fares/distances, precision issues
- `nyc_taxi_v2_duplicates_nulls.csv`: Exact/near duplicates, various NULL representations, invalid location IDs

**Online Retail (E-commerce)**
- `online_retail_v1_encoding.csv`: Special characters, whitespace, case inconsistencies, format issues
- `online_retail_v2_logic.csv`: Business logic violations (negative quantities, zero prices, future dates)

**Chicago Crimes (Public Safety)**
- `chicago_crimes_v1_geographic.csv`: Swapped lat/long, out-of-bounds coordinates, systematic missing data
- `chicago_crimes_v2_temporal.csv`: Date format chaos, IUCR/type mismatches, boolean inconsistencies

**Air Quality (Environmental/Sensors)**
- `air_quality_v1_sensors.csv`: Unit inconsistencies, impossible values (negative concentrations, AQI > 500)
- `air_quality_v2_temporal.csv`: Date format issues, coordinate precision loss, state/name mismatches

## Structure

```
.
├── raw/                      # Clean source data (not corrupted)
│   ├── nyc_taxi_jan2024.csv
│   ├── online_retail.csv
│   ├── chicago_crimes.csv
│   └── air_quality_pm25.csv
├── corrupted/                # Datasets with injected issues
│   ├── nyc_taxi_v1_timestamps_values.csv
│   ├── nyc_taxi_v2_duplicates_nulls.csv
│   ├── online_retail_v1_encoding.csv
│   ├── online_retail_v2_logic.csv
│   ├── chicago_crimes_v1_geographic.csv
│   ├── chicago_crimes_v2_temporal.csv
│   ├── air_quality_v1_sensors.csv
│   └── air_quality_v2_temporal.csv
└── corrupt_datasets.py       # Script to regenerate corrupted data
```

## Usage

### Testing Data Quality Detection

Pass any corrupted dataset to a fresh analysis session and attempt to identify issues without prior knowledge of the corruption applied.

### Regenerating Corrupted Data

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas pyarrow openpyxl requests
python3 corrupt_datasets.py
```

## Data Quality Issue Categories

Issues represented in this POC:

- **Timestamp/Date issues**: Mixed formats, future dates, timezone inconsistencies
- **Numeric issues**: Negative values where impossible, extreme outliers, precision artifacts
- **Duplicates**: Exact duplicates, near-duplicates with conflicting values
- **Missing values**: Various NULL representations (NULL, N/A, empty, etc.)
- **Encoding**: Special characters, mixed encodings, whitespace problems
- **Geographic**: Swapped coordinates, out-of-bounds values, precision loss
- **Business logic**: Constraint violations, impossible combinations
- **Referential integrity**: Invalid IDs, mismatched codes and descriptions
- **Categorical**: Inconsistent case, format variations, code mismatches

## License

The source datasets are from public open data sources:
- NYC TLC: Public domain
- UCI Online Retail: CC BY 4.0
- Chicago Data Portal: Public domain
- EPA Air Quality: Public domain
