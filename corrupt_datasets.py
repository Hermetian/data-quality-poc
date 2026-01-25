#!/usr/bin/env python3
"""
Corrupt real-world datasets with realistic data quality issues.
Each corruption is documented in detail for the POC evaluation.

DATASET SIZES:
- nyc_taxi_v1: 1,000,000 rows (1MM)
- online_retail_v1: 541,909 rows (~540K)
- chicago_crimes_v1: 500,000 rows (500K)
- air_quality_v1: 250,000 rows (250K)
- chicago_crimes_v2: 100,000 rows (100K)
- nyc_taxi_v2: 50,000 rows (50K)
- online_retail_v2: 25,000 rows (25K)
- air_quality_v2: 10,000 rows (10K)
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

os.makedirs('corrupted', exist_ok=True)

# =============================================================================
# NYC TAXI V1 - 1,000,000 ROWS
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. TIMESTAMP FORMAT CHAOS (25% of rows, ~250K affected)
#    - Original format: "2024-01-15 14:30:22"
#    - Corrupted to random mix of:
#      * "01/15/2024 14:30" (US format, no seconds)
#      * "15-01-2024 14:30:22" (European day-first)
#      * "2024/01/15 14:30:22" (slash separator)
#      * "2024-01-15T14:30:22Z" (ISO with Z suffix)
#    - Applied to: tpep_pickup_datetime column
#    - Detection hint: Parse failures, inconsistent datetime parsing
#
# 2. FUTURE TIMESTAMPS (3% of rows, ~30K affected)
#    - Pickup times set to dates in 2026 (data is from 2024)
#    - Random dates between 2026-01-01 and 2026-12-28
#    - Applied to: tpep_pickup_datetime column
#    - Detection hint: Dates beyond reasonable data collection period
#
# 3. NEGATIVE FARES (5% of rows, ~50K affected)
#    - fare_amount multiplied by -1
#    - Creates values like -15.50 instead of 15.50
#    - Applied to: fare_amount column
#    - Detection hint: Negative values in monetary field
#
# 4. NEGATIVE DISTANCES (4% of rows, ~40K affected)
#    - trip_distance multiplied by -1
#    - Creates values like -3.2 instead of 3.2
#    - Applied to: trip_distance column
#    - Detection hint: Physically impossible negative distance
#
# 5. IMPOSSIBLE PASSENGER COUNTS (3% of rows, ~30K affected)
#    - Values replaced with: -1, -2, 15, 99, or 127
#    - Normal range is 1-6
#    - Applied to: passenger_count column
#    - Detection hint: Domain violation (can't have -1 or 127 passengers)
#
# 6. FLOATING POINT PRECISION ARTIFACTS (8% of rows, ~80K affected)
#    - Added 0.0000001 to total_amount
#    - Creates values like 25.5000001 instead of 25.50
#    - Applied to: total_amount column
#    - Detection hint: Unusual decimal precision, rounding issues
#
# 7. EXTREME OUTLIER FARES (0.5% of rows, ~5K affected)
#    - fare_amount set to values between $10,000 and $999,999
#    - NYC taxi fares are typically $5-$100
#    - Applied to: fare_amount column
#    - Detection hint: Statistical outliers, implausible values
#
# 8. ZERO DISTANCE WITH HIGH FARE (2% of rows, ~20K affected)
#    - trip_distance set to 0
#    - fare_amount set to random value $50-$500
#    - Creates impossible trip (no distance but significant fare)
#    - Applied to: trip_distance, fare_amount columns
#    - Detection hint: Business logic violation
#
# 9. DROPOFF BEFORE PICKUP (1.5% of rows, ~15K affected)
#    - tpep_dropoff_datetime set to time BEFORE pickup
#    - Time travel scenario
#    - Applied to: tpep_dropoff_datetime column
#    - Detection hint: Temporal logic violation
#
# 10. LOCATION ID OUT OF RANGE (2% of rows, ~20K affected)
#     - PULocationID and DOLocationID set to 0, 999, or 9999
#     - Valid range is 1-265 (NYC taxi zones)
#     - Applied to: PULocationID, DOLocationID columns
#     - Detection hint: Referential integrity violation
#
# =============================================================================

def corrupt_taxi_v1():
    """1MM rows - Heavy timestamp and value corruption"""
    print("Creating NYC Taxi V1 (1MM rows)...")
    df = pd.read_csv('raw/nyc_taxi_1m.csv')
    print(f"  Loaded {len(df):,} rows")

    # Convert and stringify datetime for manipulation
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime']).astype(str)
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime']).astype(str)

    # 1. Timestamp format chaos (25%)
    formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%d-%m-%Y %H:%M:%S',
               '%Y/%m/%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']
    mask = np.random.random(len(df)) < 0.25
    df.loc[mask, 'tpep_pickup_datetime'] = df.loc[mask, 'tpep_pickup_datetime'].apply(
        lambda x: pd.to_datetime(x).strftime(random.choice(formats)) if pd.notna(x) and x != 'NaT' else x
    )
    print(f"    Applied timestamp format chaos to {mask.sum():,} rows")

    # 2. Future timestamps (3%)
    future_mask = np.random.random(len(df)) < 0.03
    df.loc[future_mask, 'tpep_pickup_datetime'] = [
        datetime(2026, random.randint(1,12), random.randint(1,28),
                 random.randint(0,23), random.randint(0,59)).strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(future_mask.sum())
    ]
    print(f"    Applied future timestamps to {future_mask.sum():,} rows")

    # 3. Negative fares (5%)
    neg_fare_mask = np.random.random(len(df)) < 0.05
    df.loc[neg_fare_mask, 'fare_amount'] = -abs(df.loc[neg_fare_mask, 'fare_amount'])
    print(f"    Applied negative fares to {neg_fare_mask.sum():,} rows")

    # 4. Negative distances (4%)
    neg_dist_mask = np.random.random(len(df)) < 0.04
    df.loc[neg_dist_mask, 'trip_distance'] = -abs(df.loc[neg_dist_mask, 'trip_distance'])
    print(f"    Applied negative distances to {neg_dist_mask.sum():,} rows")

    # 5. Impossible passenger counts (3%)
    pass_mask = np.random.random(len(df)) < 0.03
    df.loc[pass_mask, 'passenger_count'] = np.random.choice([-1, -2, 15, 99, 127], size=pass_mask.sum())
    print(f"    Applied impossible passenger counts to {pass_mask.sum():,} rows")

    # 6. Precision artifacts (8%)
    prec_mask = np.random.random(len(df)) < 0.08
    df.loc[prec_mask, 'total_amount'] = df.loc[prec_mask, 'total_amount'] + 0.0000001
    print(f"    Applied precision artifacts to {prec_mask.sum():,} rows")

    # 7. Extreme outlier fares (0.5%)
    outlier_mask = np.random.random(len(df)) < 0.005
    df.loc[outlier_mask, 'fare_amount'] = np.random.uniform(10000, 999999, size=outlier_mask.sum())
    print(f"    Applied extreme outlier fares to {outlier_mask.sum():,} rows")

    # 8. Zero distance with high fare (2%)
    zero_mask = np.random.random(len(df)) < 0.02
    df.loc[zero_mask, 'trip_distance'] = 0
    df.loc[zero_mask, 'fare_amount'] = np.random.uniform(50, 500, size=zero_mask.sum())
    print(f"    Applied zero distance/high fare to {zero_mask.sum():,} rows")

    # 9. Dropoff before pickup (1.5%)
    time_travel_mask = np.random.random(len(df)) < 0.015
    df.loc[time_travel_mask, 'tpep_dropoff_datetime'] = df.loc[time_travel_mask, 'tpep_pickup_datetime'].apply(
        lambda x: (pd.to_datetime(x) - timedelta(hours=random.randint(1, 5))).strftime('%Y-%m-%d %H:%M:%S')
        if pd.notna(x) and x != 'NaT' else x
    )
    print(f"    Applied dropoff before pickup to {time_travel_mask.sum():,} rows")

    # 10. Invalid location IDs (2%)
    loc_mask = np.random.random(len(df)) < 0.02
    df.loc[loc_mask, 'PULocationID'] = np.random.choice([0, 999, 9999], size=loc_mask.sum())
    df.loc[loc_mask, 'DOLocationID'] = np.random.choice([0, 888, 9999], size=loc_mask.sum())
    print(f"    Applied invalid location IDs to {loc_mask.sum():,} rows")

    df.to_csv('corrupted/nyc_taxi_v1.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/nyc_taxi_v1.csv")


# =============================================================================
# ONLINE RETAIL V1 - 541,909 ROWS (FULL DATASET)
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. SPECIAL CHARACTERS IN DESCRIPTIONS (10% of rows, ~54K affected)
#    - Prepended special chars: é, ñ, ü, ø, ß, æ, ™, ©, ®, €, £
#    - Creates: "®GLASS VASE" instead of "GLASS VASE"
#    - Applied to: Description column
#    - Detection hint: Unexpected Unicode, encoding issues
#
# 2. LEADING/TRAILING WHITESPACE (15% of rows, ~81K affected)
#    - Added random spaces and tabs: "  ", "   ", "\t", " \t "
#    - Creates: "   CANDLE HOLDER   " instead of "CANDLE HOLDER"
#    - Applied to: Description column
#    - Detection hint: String length anomalies, trimming needed
#
# 3. INCONSISTENT CASE IN COUNTRY (20% of rows, ~108K affected)
#    - Random case transformation: upper, lower, title
#    - Creates: "UNITED KINGDOM", "united kingdom", "United Kingdom"
#    - Applied to: Country column
#    - Detection hint: Duplicate countries when case-normalized
#
# 4. STOCKCODE FORMAT INCONSISTENCY (12% of rows, ~65K affected)
#    - Stripped leading zeros and lowercased
#    - Creates: "22423" -> "22423", "85123A" -> "85123a"
#    - Applied to: StockCode column
#    - Detection hint: Failed joins, inconsistent references
#
# 5. EMBEDDED QUOTES IN DESCRIPTIONS (6% of rows, ~32K affected)
#    - Wrapped descriptions in extra quotes
#    - Creates: '"GLASS VASE"' with embedded quotes
#    - Applied to: Description column
#    - Detection hint: CSV parsing issues, quote escaping
#
# 6. NULL DESCRIPTIONS (8% of rows, ~43K affected)
#    - Set Description to various null representations
#    - Values: '', 'NULL', 'N/A', 'nan', 'None', '-'
#    - Applied to: Description column
#    - Detection hint: Inconsistent null handling
#
# 7. NEGATIVE UNIT PRICES (5% of rows, ~27K affected)
#    - UnitPrice multiplied by -1
#    - Creates: -2.55 instead of 2.55
#    - Applied to: UnitPrice column
#    - Detection hint: Impossible negative price
#
# 8. INVOICE NUMBER FORMAT MIX (4% of rows, ~21K affected)
#    - Added random prefixes: INV_, #, ORDER-
#    - Creates: "INV_536365" instead of "536365"
#    - Applied to: InvoiceNo column
#    - Detection hint: Inconsistent identifier format
#
# =============================================================================

def corrupt_retail_v1():
    """~540K rows - Encoding and text corruption"""
    print("Creating Online Retail V1 (~540K rows)...")
    df = pd.read_csv('raw/online_retail.csv')
    print(f"  Loaded {len(df):,} rows")

    # 1. Special characters (10%)
    special_chars = ['é', 'ñ', 'ü', 'ø', 'ß', 'æ', '™', '©', '®', '€', '£']
    spec_mask = np.random.random(len(df)) < 0.10
    df.loc[spec_mask, 'Description'] = df.loc[spec_mask, 'Description'].apply(
        lambda x: f"{random.choice(special_chars)}{x}" if pd.notna(x) else x
    )
    print(f"    Applied special characters to {spec_mask.sum():,} rows")

    # 2. Whitespace (15%)
    spaces = ['  ', '   ', '\t', ' \t ']
    ws_mask = np.random.random(len(df)) < 0.15
    df.loc[ws_mask, 'Description'] = df.loc[ws_mask, 'Description'].apply(
        lambda x: f"{random.choice(spaces)}{x}{random.choice(spaces)}" if pd.notna(x) else x
    )
    print(f"    Applied whitespace to {ws_mask.sum():,} rows")

    # 3. Case inconsistency (20%)
    case_mask = np.random.random(len(df)) < 0.20
    df.loc[case_mask, 'Country'] = df.loc[case_mask, 'Country'].apply(
        lambda x: random.choice([str(x).upper(), str(x).lower(), str(x).title()]) if pd.notna(x) else x
    )
    print(f"    Applied case inconsistency to {case_mask.sum():,} rows")

    # 4. StockCode format (12%)
    stock_mask = np.random.random(len(df)) < 0.12
    df.loc[stock_mask, 'StockCode'] = df.loc[stock_mask, 'StockCode'].apply(
        lambda x: str(x).lstrip('0').lower() if pd.notna(x) else x
    )
    print(f"    Applied StockCode format issues to {stock_mask.sum():,} rows")

    # 5. Embedded quotes (6%)
    quote_mask = np.random.random(len(df)) < 0.06
    df.loc[quote_mask, 'Description'] = df.loc[quote_mask, 'Description'].apply(
        lambda x: f'"{x}"' if pd.notna(x) and '"' not in str(x) else x
    )
    print(f"    Applied embedded quotes to {quote_mask.sum():,} rows")

    # 6. NULL descriptions (8%)
    null_mask = np.random.random(len(df)) < 0.08
    null_values = ['', 'NULL', 'N/A', 'nan', 'None', '-']
    df.loc[null_mask, 'Description'] = np.random.choice(null_values, size=null_mask.sum())
    print(f"    Applied NULL descriptions to {null_mask.sum():,} rows")

    # 7. Negative prices (5%)
    neg_price_mask = np.random.random(len(df)) < 0.05
    df.loc[neg_price_mask, 'UnitPrice'] = -abs(df.loc[neg_price_mask, 'UnitPrice'])
    print(f"    Applied negative prices to {neg_price_mask.sum():,} rows")

    # 8. Invoice format mix (4%)
    inv_mask = np.random.random(len(df)) < 0.04
    prefixes = ['INV_', '#', 'ORDER-', 'ORD']
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df.loc[inv_mask, 'InvoiceNo'] = df.loc[inv_mask, 'InvoiceNo'].apply(
        lambda x: f"{random.choice(prefixes)}{x}"
    )
    print(f"    Applied invoice format mix to {inv_mask.sum():,} rows")

    df.to_csv('corrupted/online_retail_v1.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/online_retail_v1.csv")


# =============================================================================
# CHICAGO CRIMES V1 - 500,000 ROWS
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. SWAPPED LAT/LONG (6% of rows, ~30K affected)
#    - Latitude and longitude values swapped
#    - Chicago coords: lat ~41.8, long ~-87.6
#    - After swap: lat ~-87.6, long ~41.8 (invalid)
#    - Applied to: latitude, longitude columns
#    - Detection hint: Coordinates outside valid lat range (-90 to 90)
#
# 2. OUT OF BOUNDS COORDINATES (5% of rows, ~25K affected)
#    - Latitude set to random 30-50, longitude to random -100 to -70
#    - Chicago should be: lat 41.6-42.1, long -88.0 to -87.5
#    - Applied to: latitude, longitude columns
#    - Detection hint: Points far outside Chicago
#
# 3. ZERO COORDINATES (4% of rows, ~20K affected)
#    - Both lat and long set to exactly 0
#    - Creates points at "null island" (0,0 in Atlantic)
#    - Applied to: latitude, longitude columns
#    - Detection hint: Suspicious concentration at (0,0)
#
# 4. SYSTEMATIC MISSING DATA FOR DRUG CRIMES (35% of drug crimes, ~15K affected)
#    - For NARCOTICS crimes, location data redacted
#    - latitude/longitude set to NaN
#    - block set to "REDACTED"
#    - Applied to: latitude, longitude, block columns (filtered)
#    - Detection hint: Systematic missingness by crime type
#
# 5. INVALID DISTRICT CODES (5% of rows, ~25K affected)
#    - District set to 0, -1, 99, or 100
#    - Valid range is 1-25
#    - Applied to: district column
#    - Detection hint: Domain violation
#
# 6. INVALID WARD CODES (5% of rows, ~25K affected)
#    - Ward set to 0, -1, 99, or 100
#    - Valid range is 1-50
#    - Applied to: ward column
#    - Detection hint: Domain violation
#
# 7. LOCATION STRING CORRUPTION (8% of rows, ~40K affected)
#    - block field has extra characters appended
#    - Added: " (APPROX)", " - VERIFIED", " [REDACTED]"
#    - Applied to: block column
#    - Detection hint: Inconsistent address format
#
# 8. X/Y COORDINATE PRECISION LOSS (10% of rows, ~50K affected)
#    - x_coordinate and y_coordinate rounded to nearest 1000
#    - Loses precision for geocoding
#    - Applied to: x_coordinate, y_coordinate columns
#    - Detection hint: Clustering on round numbers
#
# =============================================================================

def corrupt_crimes_v1():
    """500K rows - Geographic corruption"""
    print("Creating Chicago Crimes V1 (500K rows)...")
    df = pd.read_csv('raw/chicago_crimes_500k.csv')
    print(f"  Loaded {len(df):,} rows")

    # 1. Swapped lat/long (6%)
    swap_mask = np.random.random(len(df)) < 0.06
    lat_temp = df.loc[swap_mask, 'latitude'].copy()
    df.loc[swap_mask, 'latitude'] = df.loc[swap_mask, 'longitude']
    df.loc[swap_mask, 'longitude'] = lat_temp
    print(f"    Swapped lat/long for {swap_mask.sum():,} rows")

    # 2. Out of bounds (5%)
    oob_mask = np.random.random(len(df)) < 0.05
    df.loc[oob_mask, 'latitude'] = np.random.uniform(30, 50, size=oob_mask.sum())
    df.loc[oob_mask, 'longitude'] = np.random.uniform(-100, -70, size=oob_mask.sum())
    print(f"    Applied out-of-bounds coords to {oob_mask.sum():,} rows")

    # 3. Zero coordinates (4%)
    zero_mask = np.random.random(len(df)) < 0.04
    df.loc[zero_mask, 'latitude'] = 0
    df.loc[zero_mask, 'longitude'] = 0
    print(f"    Applied zero coords to {zero_mask.sum():,} rows")

    # 4. Systematic drug crime redaction (35% of drug crimes)
    drug_mask = df['primary_type'].str.contains('NARCOTICS|DRUG', case=False, na=False)
    redact_mask = drug_mask & (np.random.random(len(df)) < 0.35)
    df.loc[redact_mask, 'latitude'] = np.nan
    df.loc[redact_mask, 'longitude'] = np.nan
    df.loc[redact_mask, 'block'] = 'REDACTED'
    print(f"    Redacted location for {redact_mask.sum():,} drug crime rows")

    # 5. Invalid districts (5%)
    dist_mask = np.random.random(len(df)) < 0.05
    df.loc[dist_mask, 'district'] = np.random.choice([0, -1, 99, 100], size=dist_mask.sum())
    print(f"    Applied invalid districts to {dist_mask.sum():,} rows")

    # 6. Invalid wards (5%)
    ward_mask = np.random.random(len(df)) < 0.05
    df.loc[ward_mask, 'ward'] = np.random.choice([0, -1, 99, 100], size=ward_mask.sum())
    print(f"    Applied invalid wards to {ward_mask.sum():,} rows")

    # 7. Location string corruption (8%)
    loc_str_mask = np.random.random(len(df)) < 0.08
    suffixes = [' (APPROX)', ' - VERIFIED', ' [REDACTED]', ' *', ' ???']
    df['block'] = df['block'].astype(str)
    df.loc[loc_str_mask, 'block'] = df.loc[loc_str_mask, 'block'].apply(
        lambda x: f"{x}{random.choice(suffixes)}" if pd.notna(x) else x
    )
    print(f"    Applied location string corruption to {loc_str_mask.sum():,} rows")

    # 8. X/Y precision loss (10%)
    prec_mask = np.random.random(len(df)) < 0.10
    df.loc[prec_mask, 'x_coordinate'] = (df.loc[prec_mask, 'x_coordinate'] / 1000).round() * 1000
    df.loc[prec_mask, 'y_coordinate'] = (df.loc[prec_mask, 'y_coordinate'] / 1000).round() * 1000
    print(f"    Applied X/Y precision loss to {prec_mask.sum():,} rows")

    df.to_csv('corrupted/chicago_crimes_v1.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/chicago_crimes_v1.csv")


# =============================================================================
# AIR QUALITY V1 - 250,000 ROWS
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. UNIT INCONSISTENCIES (15% of rows, ~37K affected)
#    - "Units of Measure" changed to various formats WITH VALUE CONVERSION
#    - Base unit is µg/m³. Conversions applied:
#      * 'µg/m³' -> factor 1 (no change)
#      * 'ug/m3' -> factor 1 (ASCII variant, no change)
#      * 'Micrograms/cubic meter' -> factor 1 (verbose, no change)
#      * 'mg/m3' -> factor 0.001 (milligrams)
#      * 'ng/m3' -> factor 1000 (nanograms)
#    - Applied to: Units of Measure, Arithmetic Mean, 1st Max Value columns
#    - Detection hint: Must normalize units to compare values
#    - RECOVERABLE: Convert all values back to µg/m³ using inverse factors
#
# 2. NEGATIVE CONCENTRATIONS (5% of rows, ~12K affected)
#    - Arithmetic Mean multiplied by -1
#    - PM2.5 concentration can't be negative
#    - Applied to: Arithmetic Mean column
#    - Detection hint: Physically impossible negative value
#
# 3. INVALID AQI VALUES (4% of rows, ~10K affected)
#    - AQI set to -10, 600, 999, or 1000
#    - Valid range is 0-500
#    - Applied to: AQI column
#    - Detection hint: Domain violation
#
# 4. EXTREME MAX VALUES (3% of rows, ~7K affected)
#    - "1st Max Value" set to 1000-10000
#    - Normal PM2.5 max is 0-500 µg/m³
#    - Applied to: 1st Max Value column
#    - Detection hint: Statistical outlier
#
# 5. OBSERVATION PERCENT > 100 (4% of rows, ~10K affected)
#    - Observation Percent set to 100.1-200
#    - Logically impossible (max is 100%)
#    - Applied to: Observation Percent column
#    - Detection hint: Logical impossibility
#
# 6. INVALID STATE CODES (5% of rows, ~12K affected)
#    - State Code set to 'XX', '99', '-1', 'NA'
#    - Valid codes are 2-digit FIPS codes (01-56)
#    - Applied to: State Code column
#    - Detection hint: Invalid reference code
#
# 7. METHOD NAME MISMATCH (6% of rows, ~15K affected)
#    - Method Name set to 'UNKNOWN METHOD' while keeping Method Code
#    - Creates inconsistency between code and description
#    - Applied to: Method Name column
#    - Detection hint: Code/description mismatch
#
# 8. SITE NUM FORMAT INCONSISTENCY (8% of rows, ~20K affected)
#    - Site Num format altered: leading zeros stripped, letters added
#    - Creates: "0001" -> "1", "0042" -> "42A"
#    - Applied to: Site Num column
#    - Detection hint: Inconsistent site identifiers
#
# =============================================================================

def corrupt_airquality_v1():
    """250K rows - Sensor and unit corruption"""
    print("Creating Air Quality V1 (250K rows)...")
    df = pd.read_csv('raw/air_quality_full.csv', low_memory=False)
    # Sample to 250K
    df = df.sample(n=250000, random_state=42)
    print(f"  Sampled to {len(df):,} rows")

    # 1. Unit inconsistencies (15%) - WITH PROPER VALUE CONVERSION (vectorized)
    # Base unit is µg/m³. Define conversion factors from µg/m³ to target unit.
    unit_names = ['µg/m³', 'ug/m3', 'Micrograms/cubic meter', 'mg/m3', 'ng/m3']
    unit_factors = [1, 1, 1, 0.001, 1000]

    unit_mask = np.random.random(len(df)) < 0.15
    n_affected = unit_mask.sum()

    # Randomly assign unit indices to affected rows
    unit_indices = np.random.randint(0, len(unit_names), size=n_affected)
    chosen_units = [unit_names[i] for i in unit_indices]
    chosen_factors = np.array([unit_factors[i] for i in unit_indices])

    # Apply unit labels
    df.loc[unit_mask, 'Units of Measure'] = chosen_units

    # Apply conversion factors to values
    df.loc[unit_mask, 'Arithmetic Mean'] = df.loc[unit_mask, 'Arithmetic Mean'].values * chosen_factors
    df.loc[unit_mask, '1st Max Value'] = df.loc[unit_mask, '1st Max Value'].values * chosen_factors

    print(f"    Applied unit inconsistencies (with conversion) to {n_affected:,} rows")

    # 2. Negative concentrations (5%)
    neg_mask = np.random.random(len(df)) < 0.05
    df.loc[neg_mask, 'Arithmetic Mean'] = -abs(df.loc[neg_mask, 'Arithmetic Mean'])
    print(f"    Applied negative concentrations to {neg_mask.sum():,} rows")

    # 3. Invalid AQI (4%)
    aqi_mask = np.random.random(len(df)) < 0.04
    df.loc[aqi_mask, 'AQI'] = np.random.choice([-10, 600, 999, 1000], size=aqi_mask.sum())
    print(f"    Applied invalid AQI to {aqi_mask.sum():,} rows")

    # 4. Extreme max values (3%)
    max_mask = np.random.random(len(df)) < 0.03
    df.loc[max_mask, '1st Max Value'] = np.random.uniform(1000, 10000, size=max_mask.sum())
    print(f"    Applied extreme max values to {max_mask.sum():,} rows")

    # 5. Observation percent > 100 (4%)
    df['Observation Percent'] = df['Observation Percent'].astype(float)
    obs_mask = np.random.random(len(df)) < 0.04
    df.loc[obs_mask, 'Observation Percent'] = np.random.uniform(100.1, 200, size=obs_mask.sum())
    print(f"    Applied obs percent > 100 to {obs_mask.sum():,} rows")

    # 6. Invalid state codes (5%)
    df['State Code'] = df['State Code'].astype(object)
    state_mask = np.random.random(len(df)) < 0.05
    df.loc[state_mask, 'State Code'] = np.random.choice(['XX', '99', '-1', 'NA'], size=state_mask.sum())
    print(f"    Applied invalid state codes to {state_mask.sum():,} rows")

    # 7. Method name mismatch (6%)
    method_mask = np.random.random(len(df)) < 0.06
    df.loc[method_mask, 'Method Name'] = 'UNKNOWN METHOD'
    print(f"    Applied method name mismatch to {method_mask.sum():,} rows")

    # 8. Site num format (8%)
    df['Site Num'] = df['Site Num'].astype(str)
    site_mask = np.random.random(len(df)) < 0.08
    df.loc[site_mask, 'Site Num'] = df.loc[site_mask, 'Site Num'].apply(
        lambda x: str(int(float(x))).lstrip('0') + random.choice(['', 'A', 'B', '-1']) if pd.notna(x) else x
    )
    print(f"    Applied site num format issues to {site_mask.sum():,} rows")

    df.to_csv('corrupted/air_quality_v1.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/air_quality_v1.csv")


# =============================================================================
# CHICAGO CRIMES V2 - 100,000 ROWS
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. DATE FORMAT CHAOS (30% of rows, ~30K affected)
#    - Date converted to various formats
#    - Formats: '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%d-%m-%Y %H:%M'
#    - Applied to: date column
#    - Detection hint: Multiple datetime formats
#
# 2. CRIME TYPE MISMATCH (8% of rows, ~8K affected)
#    - primary_type swapped to random different crime type
#    - IUCR code no longer matches description
#    - Applied to: primary_type column
#    - Detection hint: IUCR/type inconsistency
#
# 3. BOOLEAN FORMAT INCONSISTENCY (20% of rows, ~20K affected)
#    - arrest field set to: 'True', 'False', 'true', 'false', 'YES', 'NO', '1', '0'
#    - Applied to: arrest column
#    - Detection hint: Multiple boolean representations
#
# 4. DOMESTIC BOOLEAN INCONSISTENCY (20% of rows, ~20K affected)
#    - domestic field set to same variety of boolean values
#    - Applied to: domestic column
#    - Detection hint: Multiple boolean representations
#
# 5. CASE NUMBER DUPLICATES WITH CONFLICTS (3% of rows, ~3K affected)
#    - Same case_number, different arrest status
#    - Same case_number, different district
#    - Applied to: creates duplicate rows
#    - Detection hint: Duplicate IDs with conflicting data
#
# 6. YEAR MISMATCH (6% of rows, ~6K affected)
#    - year column doesn't match date column
#    - year set to date year - random(1-5)
#    - Applied to: year column
#    - Detection hint: Derived field inconsistency
#
# 7. FBI CODE CORRUPTION (5% of rows, ~5K affected)
#    - fbi_code set to invalid values
#    - Values: 'XX', '00', 'UNKNOWN', ''
#    - Applied to: fbi_code column
#    - Detection hint: Invalid reference code
#
# 8. UPDATED_ON BEFORE DATE (4% of rows, ~4K affected)
#    - updated_on timestamp set before crime date
#    - Logically impossible
#    - Applied to: updated_on column
#    - Detection hint: Temporal logic violation
#
# =============================================================================

def corrupt_crimes_v2():
    """100K rows - Temporal and categorical corruption"""
    print("Creating Chicago Crimes V2 (100K rows)...")
    df = pd.read_csv('raw/chicago_crimes.csv')
    print(f"  Loaded {len(df):,} rows")

    # Convert date to string for manipulation
    df['date'] = pd.to_datetime(df['date']).astype(str)

    # 1. Date format chaos (30%)
    formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%d-%m-%Y %H:%M',
               '%Y/%m/%d %H:%M:%S', '%m-%d-%Y %H:%M:%S']
    date_mask = np.random.random(len(df)) < 0.30
    df.loc[date_mask, 'date'] = df.loc[date_mask, 'date'].apply(
        lambda x: pd.to_datetime(x).strftime(random.choice(formats)) if pd.notna(x) and x != 'NaT' else x
    )
    print(f"    Applied date format chaos to {date_mask.sum():,} rows")

    # 2. Crime type mismatch (8%)
    crime_types = df['primary_type'].dropna().unique()
    type_mask = np.random.random(len(df)) < 0.08
    df.loc[type_mask, 'primary_type'] = np.random.choice(crime_types, size=type_mask.sum())
    print(f"    Applied crime type mismatch to {type_mask.sum():,} rows")

    # 3. Boolean format - arrest (20%)
    bool_values = ['True', 'False', 'true', 'false', 'YES', 'NO', '1', '0', 'Y', 'N']
    df['arrest'] = df['arrest'].astype(object)
    arrest_mask = np.random.random(len(df)) < 0.20
    df.loc[arrest_mask, 'arrest'] = np.random.choice(bool_values, size=arrest_mask.sum())
    print(f"    Applied arrest boolean chaos to {arrest_mask.sum():,} rows")

    # 4. Boolean format - domestic (20%)
    df['domestic'] = df['domestic'].astype(object)
    domestic_mask = np.random.random(len(df)) < 0.20
    df.loc[domestic_mask, 'domestic'] = np.random.choice(bool_values, size=domestic_mask.sum())
    print(f"    Applied domestic boolean chaos to {domestic_mask.sum():,} rows")

    # 5. Duplicate case numbers with conflicts (3%)
    n_dupes = int(len(df) * 0.03)
    dupe_idx = df.sample(n=n_dupes, random_state=42).index
    dupes = df.loc[dupe_idx].copy()
    dupes['arrest'] = dupes['arrest'].apply(lambda x: 'False' if str(x).lower() in ['true', '1', 'yes', 'y'] else 'True')
    dupes['district'] = pd.to_numeric(dupes['district'], errors='coerce').fillna(0).astype(int) + 1
    df = pd.concat([df, dupes], ignore_index=True)
    print(f"    Added {n_dupes:,} duplicate case numbers with conflicts")

    # 6. Year mismatch (6%)
    year_mask = np.random.random(len(df)) < 0.06
    df.loc[year_mask, 'year'] = df.loc[year_mask, 'year'].apply(
        lambda x: int(x) - random.randint(1, 5) if pd.notna(x) else x
    )
    print(f"    Applied year mismatch to {year_mask.sum():,} rows")

    # 7. FBI code corruption (5%)
    df['fbi_code'] = df['fbi_code'].astype(object)
    fbi_mask = np.random.random(len(df)) < 0.05
    df.loc[fbi_mask, 'fbi_code'] = np.random.choice(['XX', '00', 'UNKNOWN', ''], size=fbi_mask.sum())
    print(f"    Applied FBI code corruption to {fbi_mask.sum():,} rows")

    # 8. Updated before date (4%)
    update_mask = np.random.random(len(df)) < 0.04
    df.loc[update_mask, 'updated_on'] = df.loc[update_mask, 'date'].apply(
        lambda x: (pd.to_datetime(x) - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d %H:%M:%S')
        if pd.notna(x) and x != 'NaT' else x
    )
    print(f"    Applied updated_on before date to {update_mask.sum():,} rows")

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv('corrupted/chicago_crimes_v2.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/chicago_crimes_v2.csv")


# =============================================================================
# NYC TAXI V2 - 50,000 ROWS
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. EXACT DUPLICATES (5% of dataset, ~2.5K affected)
#    - Complete row duplicates
#    - Same trip appearing multiple times
#    - Applied to: entire rows
#    - Detection hint: Duplicate row detection
#
# 2. NEAR DUPLICATES (4% of dataset, ~2K affected)
#    - Same trip, slightly different amounts
#    - total_amount varied by ±$0.50
#    - tip_amount varied by ±$0.25
#    - Applied to: total_amount, tip_amount columns
#    - Detection hint: Similar rows with minor variations
#
# 3. NULL VALUE VARIATIONS (8% per column, ~4K affected each)
#    - Various null representations: '', 'NULL', 'None', 'N/A', 'null', 'NaN', '-'
#    - Applied to: passenger_count, RatecodeID, payment_type, congestion_surcharge
#    - Detection hint: Inconsistent null handling
#
# 4. INVALID LOCATION IDS (6% of rows, ~3K affected)
#    - PULocationID set to 0, -1, 999, 9999
#    - DOLocationID set to 0, -1, 888, 9999
#    - Valid range: 1-265
#    - Applied to: PULocationID, DOLocationID columns
#    - Detection hint: Invalid foreign key references
#
# 5. VENDOR ID FORMAT INCONSISTENCY (7% of rows, ~3.5K affected)
#    - VendorID converted to various formats
#    - 1 -> '1', 'CMT', 'Creative Mobile', 'one'
#    - 2 -> '2', 'VTS', 'VeriFone', 'two'
#    - Applied to: VendorID column
#    - Detection hint: Inconsistent categorical encoding
#
# 6. PAYMENT TYPE TEXT MIXING (6% of rows, ~3K affected)
#    - Payment type mixed numeric/text
#    - 1 -> 'Credit', 2 -> 'Cash', 3 -> 'No charge', 4 -> 'Dispute'
#    - Applied to: payment_type column
#    - Detection hint: Mixed encoding scheme
#
# 7. RATECODE CORRUPTION (5% of rows, ~2.5K affected)
#    - RatecodeID set to invalid values: 0, 7, 99
#    - Valid range: 1-6
#    - Applied to: RatecodeID column
#    - Detection hint: Domain violation
#
# 8. STORE AND FWD FLAG INCONSISTENCY (10% of rows, ~5K affected)
#    - Flag set to various yes/no formats
#    - 'Y', 'N', 'Yes', 'No', 'YES', 'NO', '1', '0', 'true', 'false'
#    - Applied to: store_and_fwd_flag column
#    - Detection hint: Boolean format chaos
#
# =============================================================================

def corrupt_taxi_v2():
    """50K rows - Duplicates and null corruption"""
    print("Creating NYC Taxi V2 (50K rows)...")
    df = pd.read_csv('raw/nyc_taxi_jan2024.csv')
    df = df.sample(n=50000, random_state=42)
    print(f"  Sampled to {len(df):,} rows")

    # 1. Exact duplicates (5%)
    n_exact = int(len(df) * 0.05)
    exact_dupes = df.sample(n=n_exact, random_state=42)
    df = pd.concat([df, exact_dupes], ignore_index=True)
    print(f"    Added {n_exact:,} exact duplicates")

    # 2. Near duplicates (4%)
    n_near = int(len(df) * 0.04)
    near_dupes = df.sample(n=n_near, random_state=43).copy()
    near_dupes['total_amount'] = near_dupes['total_amount'] + np.random.uniform(-0.5, 0.5, size=n_near)
    near_dupes['tip_amount'] = near_dupes['tip_amount'] + np.random.uniform(-0.25, 0.25, size=n_near)
    df = pd.concat([df, near_dupes], ignore_index=True)
    print(f"    Added {n_near:,} near duplicates")

    # 3. Null variations (8% per column)
    null_values = ['', 'NULL', 'None', 'N/A', 'null', 'NaN', '-', '.']
    for col in ['passenger_count', 'RatecodeID', 'payment_type', 'congestion_surcharge']:
        df[col] = df[col].astype(object)
        null_mask = np.random.random(len(df)) < 0.08
        df.loc[null_mask, col] = np.random.choice(null_values, size=null_mask.sum())
        print(f"    Applied null variations to {col}: {null_mask.sum():,} rows")

    # 4. Invalid location IDs (6%)
    loc_mask = np.random.random(len(df)) < 0.06
    df.loc[loc_mask, 'PULocationID'] = np.random.choice([0, -1, 999, 9999], size=loc_mask.sum())
    df.loc[loc_mask, 'DOLocationID'] = np.random.choice([0, -1, 888, 9999], size=loc_mask.sum())
    print(f"    Applied invalid location IDs to {loc_mask.sum():,} rows")

    # 5. Vendor ID format (7%)
    df['VendorID'] = df['VendorID'].astype(object)
    vendor_mask = np.random.random(len(df)) < 0.07
    vendor_map = {1: ['1', 'CMT', 'Creative Mobile', 'one'],
                  2: ['2', 'VTS', 'VeriFone', 'two']}
    df.loc[vendor_mask, 'VendorID'] = df.loc[vendor_mask, 'VendorID'].apply(
        lambda x: random.choice(vendor_map.get(int(x) if pd.notna(x) and x != '' else 0, [x])) if pd.notna(x) else x
    )
    print(f"    Applied vendor ID format issues to {vendor_mask.sum():,} rows")

    # 6. Payment type text mixing (6%)
    pay_mask = np.random.random(len(df)) < 0.06
    pay_map = {1: 'Credit', 2: 'Cash', 3: 'No charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided'}
    df.loc[pay_mask, 'payment_type'] = df.loc[pay_mask, 'payment_type'].apply(
        lambda x: pay_map.get(int(x) if pd.notna(x) and str(x).isdigit() else 0, x)
    )
    print(f"    Applied payment type text mixing to {pay_mask.sum():,} rows")

    # 7. Ratecode corruption (5%)
    rate_mask = np.random.random(len(df)) < 0.05
    df.loc[rate_mask, 'RatecodeID'] = np.random.choice([0, 7, 99, -1], size=rate_mask.sum())
    print(f"    Applied ratecode corruption to {rate_mask.sum():,} rows")

    # 8. Store and fwd flag inconsistency (10%)
    flag_values = ['Y', 'N', 'Yes', 'No', 'YES', 'NO', '1', '0', 'true', 'false']
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype(object)
    flag_mask = np.random.random(len(df)) < 0.10
    df.loc[flag_mask, 'store_and_fwd_flag'] = np.random.choice(flag_values, size=flag_mask.sum())
    print(f"    Applied store_and_fwd_flag inconsistency to {flag_mask.sum():,} rows")

    df = df.sample(frac=1, random_state=44).reset_index(drop=True)
    df.to_csv('corrupted/nyc_taxi_v2.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/nyc_taxi_v2.csv")


# =============================================================================
# ONLINE RETAIL V2 - 25,000 ROWS
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. NEGATIVE QUANTITIES WITHOUT CANCELLATION (8% of rows, ~2K affected)
#    - Quantity set to negative for non-cancelled orders
#    - Cancelled orders start with 'C' in InvoiceNo
#    - Applied to: Quantity column
#    - Detection hint: Business logic violation
#
# 2. ZERO PRICES (6% of rows, ~1.5K affected)
#    - UnitPrice set to 0
#    - Valid transactions should have price > 0
#    - Applied to: UnitPrice column
#    - Detection hint: Invalid pricing
#
# 3. EXTREME PRICES (3% of rows, ~750 affected)
#    - UnitPrice set to 10,000-999,999
#    - Normal range is typically 0.01-500
#    - Applied to: UnitPrice column
#    - Detection hint: Statistical outlier
#
# 4. FUTURE INVOICE DATES (4% of rows, ~1K affected)
#    - InvoiceDate set to 2026
#    - Data is from 2010-2011
#    - Applied to: InvoiceDate column
#    - Detection hint: Temporal impossibility
#
# 5. INVALID CUSTOMER IDS (7% of rows, ~1.75K affected)
#    - CustomerID set to 'GUEST', 'UNKNOWN', 'TEST', '-1', 'NULL', '0'
#    - Should be numeric
#    - Applied to: CustomerID column
#    - Detection hint: Invalid foreign key
#
# 6. QUANTITY/PRICE MATH ERRORS (5% of rows, ~1.25K affected)
#    - Introduced calculated "LineTotal" column
#    - LineTotal != Quantity * UnitPrice (off by random amount)
#    - Applied to: new LineTotal column
#    - Detection hint: Calculation verification failure
#
# 7. DUPLICATE INVOICES WITH DIFFERENT ITEMS (4% of rows, ~1K affected)
#    - Same InvoiceNo assigned to different rows
#    - Creates referential ambiguity
#    - Applied to: InvoiceNo column
#    - Detection hint: Duplicate key with different values
#
# 8. COUNTRY CODE MIX (6% of rows, ~1.5K affected)
#    - Country replaced with ISO codes mixed with names
#    - 'United Kingdom' -> 'UK', 'GB', 'GBR'
#    - 'France' -> 'FR', 'FRA'
#    - Applied to: Country column
#    - Detection hint: Inconsistent country encoding
#
# =============================================================================

def corrupt_retail_v2():
    """25K rows - Business logic corruption"""
    print("Creating Online Retail V2 (25K rows)...")
    df = pd.read_csv('raw/online_retail.csv')
    df = df.sample(n=25000, random_state=42)
    print(f"  Sampled to {len(df):,} rows")

    # 1. Negative quantities for non-cancelled (8%)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    invoice_not_cancel = ~df['InvoiceNo'].str.startswith('C')
    neg_qty_mask = (np.random.random(len(df)) < 0.08) & invoice_not_cancel
    df.loc[neg_qty_mask, 'Quantity'] = -abs(df.loc[neg_qty_mask, 'Quantity'])
    print(f"    Applied negative quantities to {neg_qty_mask.sum():,} rows")

    # 2. Zero prices (6%)
    zero_price_mask = np.random.random(len(df)) < 0.06
    df.loc[zero_price_mask, 'UnitPrice'] = 0
    print(f"    Applied zero prices to {zero_price_mask.sum():,} rows")

    # 3. Extreme prices (3%)
    extreme_mask = np.random.random(len(df)) < 0.03
    df.loc[extreme_mask, 'UnitPrice'] = np.random.uniform(10000, 999999, size=extreme_mask.sum())
    print(f"    Applied extreme prices to {extreme_mask.sum():,} rows")

    # 4. Future dates (4%)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).astype(str)
    future_mask = np.random.random(len(df)) < 0.04
    future_dates = pd.date_range(start='2026-01-01', periods=future_mask.sum(), freq='h').strftime('%Y-%m-%d %H:%M:%S')
    df.loc[future_mask, 'InvoiceDate'] = list(future_dates)
    print(f"    Applied future dates to {future_mask.sum():,} rows")

    # 5. Invalid customer IDs (7%)
    df['CustomerID'] = df['CustomerID'].astype(object)
    cust_mask = np.random.random(len(df)) < 0.07
    invalid_ids = ['GUEST', 'UNKNOWN', 'TEST', '-1', 'NULL', '0']
    df.loc[cust_mask, 'CustomerID'] = np.random.choice(invalid_ids, size=cust_mask.sum())
    print(f"    Applied invalid customer IDs to {cust_mask.sum():,} rows")

    # 6. Math errors - add LineTotal with errors (5%)
    df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    math_mask = np.random.random(len(df)) < 0.05
    df.loc[math_mask, 'LineTotal'] = df.loc[math_mask, 'LineTotal'] + np.random.uniform(-10, 10, size=math_mask.sum())
    print(f"    Applied math errors to {math_mask.sum():,} rows")

    # 7. Duplicate invoices (4%)
    dupe_inv_mask = np.random.random(len(df)) < 0.04
    existing_invoices = df.loc[~dupe_inv_mask, 'InvoiceNo'].dropna().unique()
    if len(existing_invoices) > 0:
        df.loc[dupe_inv_mask, 'InvoiceNo'] = np.random.choice(existing_invoices, size=dupe_inv_mask.sum())
    print(f"    Applied duplicate invoices to {dupe_inv_mask.sum():,} rows")

    # 8. Country code mix (6%)
    country_map = {
        'United Kingdom': ['UK', 'GB', 'GBR', 'Britain'],
        'France': ['FR', 'FRA', 'FRANCE'],
        'Germany': ['DE', 'DEU', 'GERMANY'],
        'Spain': ['ES', 'ESP', 'SPAIN'],
        'Netherlands': ['NL', 'NLD', 'NETHERLANDS']
    }
    country_mask = np.random.random(len(df)) < 0.06
    df.loc[country_mask, 'Country'] = df.loc[country_mask, 'Country'].apply(
        lambda x: random.choice(country_map.get(x, [x])) if pd.notna(x) else x
    )
    print(f"    Applied country code mix to {country_mask.sum():,} rows")

    df.to_csv('corrupted/online_retail_v2.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/online_retail_v2.csv")


# =============================================================================
# AIR QUALITY V2 - 10,000 ROWS
# =============================================================================
#
# CORRUPTION MANIFEST:
#
# 1. DATE FORMAT CHAOS (25% of rows, ~2.5K affected)
#    - Date Local converted to various formats
#    - Formats: '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d'
#    - Applied to: Date Local column
#    - Detection hint: Multiple date formats
#
# 2. FUTURE DATES (3% of rows, ~300 affected)
#    - Date set to 2026
#    - Data is from 2023
#    - Applied to: Date Local column
#    - Detection hint: Temporal impossibility
#
# 3. COORDINATE PRECISION LOSS (12% of rows, ~1.2K affected)
#    - Latitude and Longitude rounded to 1 decimal
#    - Loses precision for location matching
#    - Applied to: Latitude, Longitude columns
#    - Detection hint: Clustering on round coordinates
#
# 4. STATE NAME/CODE MISMATCH (8% of rows, ~800 affected)
#    - State Name swapped to different state
#    - State Code no longer matches State Name
#    - Applied to: State Name column
#    - Detection hint: Reference inconsistency
#
# 5. EXACT DUPLICATES (4% of dataset, ~400 affected)
#    - Complete row duplicates
#    - Applied to: entire rows
#    - Detection hint: Duplicate records
#
# 6. NEAR DUPLICATE READINGS (5% of dataset, ~500 affected)
#    - Same site, same day, slightly different readings
#    - Arithmetic Mean varied by ±10%
#    - AQI varied by ±5
#    - Applied to: Arithmetic Mean, AQI columns
#    - Detection hint: Conflicting readings
#
# 7. CBSA NAME INCONSISTENCY (10% of rows, ~1K affected)
#    - CBSA Name case/format variations
#    - 'Los Angeles-Long Beach-Anaheim, CA' vs 'LOS ANGELES LONG BEACH ANAHEIM CA'
#    - Applied to: CBSA Name column
#    - Detection hint: String normalization needed
#
# 8. COUNTY CODE/NAME MISMATCH (6% of rows, ~600 affected)
#    - County Name swapped to different county
#    - County Code no longer matches
#    - Applied to: County Name column
#    - Detection hint: Reference inconsistency
#
# =============================================================================

def corrupt_airquality_v2():
    """10K rows - Temporal and geographic corruption"""
    print("Creating Air Quality V2 (10K rows)...")
    df = pd.read_csv('raw/air_quality_full.csv', low_memory=False)
    df = df.sample(n=10000, random_state=42)
    print(f"  Sampled to {len(df):,} rows")

    # 1. Date format chaos (25%)
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
    date_mask = np.random.random(len(df)) < 0.25
    df.loc[date_mask, 'Date Local'] = df.loc[date_mask, 'Date Local'].apply(
        lambda x: pd.to_datetime(x).strftime(random.choice(formats)) if pd.notna(x) else x
    )
    print(f"    Applied date format chaos to {date_mask.sum():,} rows")

    # 2. Future dates (3%)
    future_mask = np.random.random(len(df)) < 0.03
    future_dates = pd.date_range(start='2026-01-01', periods=future_mask.sum(), freq='D').strftime('%Y-%m-%d')
    df.loc[future_mask, 'Date Local'] = list(future_dates)
    print(f"    Applied future dates to {future_mask.sum():,} rows")

    # 3. Coordinate precision loss (12%)
    coord_mask = np.random.random(len(df)) < 0.12
    df.loc[coord_mask, 'Latitude'] = df.loc[coord_mask, 'Latitude'].apply(
        lambda x: round(x, 1) if pd.notna(x) else x
    )
    df.loc[coord_mask, 'Longitude'] = df.loc[coord_mask, 'Longitude'].apply(
        lambda x: round(x, 1) if pd.notna(x) else x
    )
    print(f"    Applied coordinate precision loss to {coord_mask.sum():,} rows")

    # 4. State name/code mismatch (8%)
    states = df['State Name'].dropna().unique()
    state_mask = np.random.random(len(df)) < 0.08
    df.loc[state_mask, 'State Name'] = np.random.choice(states, size=state_mask.sum())
    print(f"    Applied state name mismatch to {state_mask.sum():,} rows")

    # 5. Exact duplicates (4%)
    n_dupes = int(len(df) * 0.04)
    dupes = df.sample(n=n_dupes, random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)
    print(f"    Added {n_dupes:,} exact duplicates")

    # 6. Near duplicates (5%)
    n_near = int(len(df) * 0.05)
    near_dupes = df.sample(n=n_near, random_state=43).copy()
    near_dupes['Arithmetic Mean'] = near_dupes['Arithmetic Mean'] * np.random.uniform(0.9, 1.1, size=n_near)
    near_dupes['AQI'] = near_dupes['AQI'] + np.random.randint(-5, 5, size=n_near)
    df = pd.concat([df, near_dupes], ignore_index=True)
    print(f"    Added {n_near:,} near duplicates")

    # 7. CBSA name inconsistency (10%)
    cbsa_mask = np.random.random(len(df)) < 0.10
    df.loc[cbsa_mask, 'CBSA Name'] = df.loc[cbsa_mask, 'CBSA Name'].apply(
        lambda x: str(x).upper().replace('-', ' ').replace(',', '') if pd.notna(x) and random.random() > 0.5 else (
            str(x).lower() if pd.notna(x) else x
        )
    )
    print(f"    Applied CBSA name inconsistency to {cbsa_mask.sum():,} rows")

    # 8. County name mismatch (6%)
    counties = df['County Name'].dropna().unique()
    county_mask = np.random.random(len(df)) < 0.06
    df.loc[county_mask, 'County Name'] = np.random.choice(counties, size=county_mask.sum())
    print(f"    Applied county name mismatch to {county_mask.sum():,} rows")

    df = df.sample(frac=1, random_state=45).reset_index(drop=True)
    df.to_csv('corrupted/air_quality_v2.csv', index=False)
    print(f"  Saved {len(df):,} rows to corrupted/air_quality_v2.csv")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA QUALITY POC - Corrupting Real Datasets")
    print("=" * 70)
    print()
    print("Target sizes:")
    print("  - nyc_taxi_v1:      1,000,000 rows")
    print("  - online_retail_v1:   541,909 rows")
    print("  - chicago_crimes_v1:  500,000 rows")
    print("  - air_quality_v1:     250,000 rows")
    print("  - chicago_crimes_v2:  100,000 rows")
    print("  - nyc_taxi_v2:         50,000 rows")
    print("  - online_retail_v2:    25,000 rows")
    print("  - air_quality_v2:      10,000 rows")
    print()
    print("=" * 70)
    print()

    # Check raw files
    required = ['raw/nyc_taxi_1m.csv', 'raw/online_retail.csv',
                'raw/chicago_crimes_500k.csv', 'raw/air_quality_full.csv',
                'raw/chicago_crimes.csv', 'raw/nyc_taxi_jan2024.csv']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing raw files: {missing}")
        exit(1)

    corrupt_taxi_v1()
    print()
    corrupt_retail_v1()
    print()
    corrupt_crimes_v1()
    print()
    corrupt_airquality_v1()
    print()
    corrupt_crimes_v2()
    print()
    corrupt_taxi_v2()
    print()
    corrupt_retail_v2()
    print()
    corrupt_airquality_v2()
    print()

    print("=" * 70)
    print("CORRUPTION COMPLETE")
    print("=" * 70)
