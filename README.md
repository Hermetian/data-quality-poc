# Data Quality POC

A proof-of-concept demonstrating data quality detection capabilities using real-world open datasets corrupted with realistic production data issues.

## Purpose

This POC demonstrates:
1. Real-world data corrupted with common production issues
2. Blind detection of data quality problems from raw data alone
3. Range of dataset sizes from 10K to 1MM rows

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

### NYC Taxi V1 (900K rows) - Timestamps & Impossible Values

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Timestamp format chaos | 25% | ~250K | Mixed formats: `2024-01-15 14:30:22`, `01/15/2024 14:30`, `15-01-2024 14:30:22`, `2024-01-15T14:30:22Z` |
| Future timestamps | 3% | ~30K | Pickup dates in 2026 (data is from 2024) |
| Negative fares | 5% | ~50K | `fare_amount` multiplied by -1 |
| Negative distances | 4% | ~40K | `trip_distance` multiplied by -1 |
| Impossible passengers | 3% | ~30K | Values: -1, -2, 15, 99, 127 (valid: 1-6) |
| Precision artifacts | 8% | ~80K | Added 0.0000001 to `total_amount` |
| Extreme outlier fares | 0.5% | ~5K | Fares $10,000-$999,999 |
| Zero distance high fare | 2% | ~20K | `trip_distance=0` but `fare_amount=$50-500` |
| Dropoff before pickup | 1.5% | ~15K | `tpep_dropoff_datetime` before `tpep_pickup_datetime` |
| Invalid location IDs | 2% | ~20K | Values: 0, 888, 999, 9999 (valid: 1-265) |

---

### Online Retail V1 (542K rows) - Encoding & Text Issues

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Special characters | 10% | ~54K | Prepended: é, ñ, ü, ø, ß, æ, ™, ©, ®, €, £ |
| Whitespace | 15% | ~81K | Leading/trailing: `"  ", "   ", "\t", " \t "` |
| Case inconsistency | 20% | ~108K | Country: `UNITED KINGDOM`, `united kingdom`, `United Kingdom` |
| StockCode format | 12% | ~65K | Stripped leading zeros, lowercased |
| Embedded quotes | 6% | ~32K | `'"GLASS VASE"'` with extra quotes |
| NULL descriptions | 8% | ~43K | Values: `''`, `'NULL'`, `'N/A'`, `'nan'`, `'None'`, `'-'` |
| Negative prices | 5% | ~27K | `UnitPrice` multiplied by -1 |
| Invoice format mix | 4% | ~22K | Prefixes: `INV_`, `#`, `ORDER-`, `ORD` |

---

### Chicago Crimes V1 (393K rows) - Geographic Corruption

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Swapped lat/long | 6% | ~30K | Latitude and longitude swapped (lat becomes ~-87) |
| Out of bounds | 5% | ~25K | Random lat 30-50, long -100 to -70 (Chicago: 41.6-42.1, -88 to -87.5) |
| Zero coordinates | 4% | ~20K | Both lat and long = 0 ("null island") |
| Drug crime redaction | 35% of drugs | ~5K | NARCOTICS crimes: lat/long=NaN, block="REDACTED" |
| Invalid districts | 5% | ~25K | Values: 0, -1, 99, 100 (valid: 1-25) |
| Invalid wards | 5% | ~25K | Values: 0, -1, 99, 100 (valid: 1-50) |
| Location string corruption | 8% | ~40K | Suffixes: ` (APPROX)`, ` - VERIFIED`, ` [REDACTED]` |
| X/Y precision loss | 10% | ~50K | Rounded to nearest 1000 |

---

### Air Quality V1 (250K rows) - Sensor & Unit Issues

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Unit inconsistencies | 15% | ~37K | `µg/m³`, `ug/m3`, `ppm`, `ppb`, `mg/m3`, `Micrograms/cubic meter` |
| Negative concentrations | 5% | ~12K | `Arithmetic Mean` multiplied by -1 |
| Invalid AQI | 4% | ~10K | Values: -10, 600, 999, 1000 (valid: 0-500) |
| Extreme max values | 3% | ~7K | `1st Max Value` = 1000-10000 |
| Observation % > 100 | 4% | ~10K | Values: 100.1-200 (max should be 100) |
| Invalid state codes | 5% | ~12K | Values: `XX`, `99`, `-1`, `NA` |
| Method name mismatch | 6% | ~15K | `Method Name` = `UNKNOWN METHOD` |
| Site num format | 8% | ~20K | Stripped zeros, added letters: `0042` → `42A` |

---

### Chicago Crimes V2 (103K rows) - Temporal & Categorical

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Date format chaos | 30% | ~30K | `%Y-%m-%d %H:%M:%S`, `%m/%d/%Y %I:%M:%S %p`, `%d-%m-%Y %H:%M` |
| Crime type mismatch | 8% | ~8K | `primary_type` doesn't match IUCR code |
| Boolean chaos (arrest) | 20% | ~20K | `True`, `False`, `YES`, `NO`, `1`, `0`, `Y`, `N` |
| Boolean chaos (domestic) | 20% | ~20K | Same variety of boolean formats |
| Duplicate case numbers | 3% | ~3K | Same `case_number`, different `arrest`/`district` |
| Year mismatch | 6% | ~6K | `year` column doesn't match `date` |
| FBI code corruption | 5% | ~5K | Values: `XX`, `00`, `UNKNOWN`, `''` |
| Updated before date | 4% | ~4K | `updated_on` timestamp before crime `date` |

---

### NYC Taxi V2 (55K rows) - Duplicates & Nulls

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Exact duplicates | 5% | ~2.5K | Complete row duplicates |
| Near duplicates | 4% | ~2K | Same trip, `total_amount` ±$0.50, `tip_amount` ±$0.25 |
| NULL variations | 8% each | ~4K each | `''`, `NULL`, `None`, `N/A`, `null`, `NaN`, `-` |
| Invalid location IDs | 6% | ~3K | Values: 0, -1, 888, 999, 9999 |
| Vendor ID format | 7% | ~3.5K | `1` → `CMT`, `Creative Mobile`, `one` |
| Payment type mix | 6% | ~3K | `1` → `Credit`, `2` → `Cash` |
| Ratecode corruption | 5% | ~2.5K | Values: 0, 7, 99, -1 (valid: 1-6) |
| Store/fwd flag chaos | 10% | ~5K | `Y`, `N`, `Yes`, `No`, `1`, `0`, `true`, `false` |

---

### Online Retail V2 (25K rows) - Business Logic

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Negative qty (non-cancel) | 8% | ~2K | Negative `Quantity` for non-cancelled orders |
| Zero prices | 6% | ~1.5K | `UnitPrice = 0` |
| Extreme prices | 3% | ~750 | `UnitPrice` = $10,000-$999,999 |
| Future dates | 4% | ~1K | `InvoiceDate` in 2026 (data from 2010-2011) |
| Invalid customer IDs | 7% | ~1.75K | `GUEST`, `UNKNOWN`, `TEST`, `-1`, `NULL` |
| Math errors | 5% | ~1.25K | `LineTotal` ≠ `Quantity * UnitPrice` |
| Duplicate invoices | 4% | ~1K | Same `InvoiceNo` for different items |
| Country code mix | 6% | ~1.5K | `United Kingdom` → `UK`, `GB`, `GBR` |

---

### Air Quality V2 (11K rows) - Temporal & Geographic

| Issue | % Affected | Rows | Description |
|-------|-----------|------|-------------|
| Date format chaos | 25% | ~2.5K | `%Y-%m-%d`, `%m/%d/%Y`, `%d-%m-%Y`, `%Y/%m/%d` |
| Future dates | 3% | ~300 | Dates in 2026 (data from 2023) |
| Coordinate precision loss | 12% | ~1.2K | Lat/long rounded to 1 decimal |
| State name mismatch | 8% | ~800 | `State Name` doesn't match `State Code` |
| Exact duplicates | 4% | ~400 | Complete row duplicates |
| Near duplicates | 5% | ~500 | Same site/day, readings ±10% |
| CBSA name inconsistency | 10% | ~1K | `Los Angeles...` vs `LOS ANGELES...` |
| County name mismatch | 6% | ~600 | `County Name` doesn't match `County Code` |

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
