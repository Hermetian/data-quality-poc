# Data Quality POC

A proof-of-concept demonstrating data quality detection capabilities. This repo contains synthetic datasets with realistic data quality issues commonly found in production environments.

## Purpose

This POC demonstrates:
1. How production data can become corrupted in various ways
2. Identifying data quality issues from raw data alone

## Datasets

### Domains

| Domain | Clean Rows | Description |
|--------|-----------|-------------|
| E-commerce Orders | 2,000 | Order transactions with products, customers, pricing |
| Financial Transactions | 5,000 | Bank/payment transactions with amounts, timestamps |
| IoT Sensors | 10,000 | Time-series sensor readings (temp, humidity, pressure) |
| Healthcare Visits | 3,000 | Patient visit records with diagnosis codes |

### Corrupted Versions

Each domain has 2 corrupted versions with different issues:

**E-commerce Orders**
- `v1_mixed_formats`: Mixed date formats, type inconsistencies, various null representations
- `v2_duplicates_integrity`: Exact/near duplicates, referential integrity violations

**Financial Transactions**
- `v1_outliers`: Extreme outliers, impossible values (negative amounts, future dates), precision issues
- `v2_formats`: Mixed currencies, timezone chaos, scientific notation, decimal separator issues

**IoT Sensors**
- `v1_units_truncation`: Mixed units (Celsius/Fahrenheit), truncated records, ID format inconsistency
- `v2_timeseries`: Timestamp ordering issues, gaps in data, duplicate timestamps

**Healthcare Visits**
- `v1_encoding`: Mixed character encodings, special characters, whitespace issues
- `v2_logic`: Business logic violations, mismatched codes/descriptions, impossible combinations

## Structure

```
.
├── clean/                    # Ground truth datasets
│   ├── ecommerce_orders.csv
│   ├── financial_transactions.csv
│   ├── healthcare_visits.csv
│   └── iot_sensors.csv
├── corrupted/                # Datasets with injected issues
│   ├── ecommerce_orders_v1_mixed_formats.csv
│   ├── ecommerce_orders_v2_duplicates_integrity.csv
│   ├── financial_transactions_v1_outliers.csv
│   ├── financial_transactions_v2_formats.csv
│   ├── healthcare_visits_v1_encoding.csv
│   ├── healthcare_visits_v2_logic.csv
│   ├── iot_sensors_v1_units_truncation.csv
│   └── iot_sensors_v2_timeseries.csv
└── generate_datasets.py      # Script to regenerate datasets
```

## Usage

### Regenerate Datasets

```bash
python3 generate_datasets.py
```

### Testing Data Quality Detection

Pass any corrupted dataset to a fresh analysis session and attempt to identify issues without prior knowledge of the corruption applied.

## Data Quality Issue Categories

Issues represented in this POC:

- **Format inconsistencies**: Date formats, number formats, ID patterns
- **Type issues**: Strings where numbers expected, mixed types
- **Missing values**: NULL, empty, N/A, various representations
- **Duplicates**: Exact duplicates, near-duplicates
- **Referential integrity**: Invalid foreign keys
- **Outliers**: Statistical outliers, impossible values
- **Encoding**: UTF-8/Latin-1 mixing, special characters
- **Business logic**: Invalid combinations, constraint violations
- **Time series**: Gaps, ordering, duplicate timestamps
- **Unit confusion**: Mixed units without indication
