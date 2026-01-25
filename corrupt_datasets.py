#!/usr/bin/env python3
"""
Corrupt real-world datasets with realistic data quality issues.
These simulate problems commonly found in production data pipelines.
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
# NYC TAXI DATA CORRUPTIONS
# =============================================================================

def corrupt_taxi_v1():
    """
    Corruption: Timestamp chaos, impossible values, precision issues
    - Mixed timezone representations
    - Future pickup times
    - Negative fares and distances
    - Floating point precision artifacts
    """
    print("Corrupting NYC Taxi v1: timestamps, impossible values...")
    df = pd.read_csv('raw/nyc_taxi_jan2024.csv')

    # Convert datetime columns and then to string for corruption
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Convert to string first for mixed format corruption
    df['tpep_pickup_datetime'] = df['tpep_pickup_datetime'].astype(str)
    df['tpep_dropoff_datetime'] = df['tpep_dropoff_datetime'].astype(str)

    # Mixed timestamp formats (20% of rows)
    formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%d-%m-%Y %H:%M:%S',
               '%Y/%m/%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']
    mask = np.random.random(len(df)) < 0.20
    df.loc[mask, 'tpep_pickup_datetime'] = df.loc[mask, 'tpep_pickup_datetime'].apply(
        lambda x: pd.to_datetime(x).strftime(random.choice(formats)) if pd.notna(x) and x != 'NaT' else x
    )

    # Future timestamps (3% of rows) - dates in 2026
    future_mask = np.random.random(len(df)) < 0.03
    df.loc[future_mask, 'tpep_pickup_datetime'] = [
        datetime(2026, random.randint(1,12), random.randint(1,28),
                 random.randint(0,23), random.randint(0,59)).strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(future_mask.sum())
    ]

    # Negative fares (5% of rows)
    neg_fare_mask = np.random.random(len(df)) < 0.05
    df.loc[neg_fare_mask, 'fare_amount'] = -abs(df.loc[neg_fare_mask, 'fare_amount'])

    # Negative distances (4% of rows)
    neg_dist_mask = np.random.random(len(df)) < 0.04
    df.loc[neg_dist_mask, 'trip_distance'] = -abs(df.loc[neg_dist_mask, 'trip_distance'])

    # Impossible passenger counts (negative or >10)
    pass_mask = np.random.random(len(df)) < 0.03
    df.loc[pass_mask, 'passenger_count'] = np.random.choice([-1, -2, 15, 99, 127], size=pass_mask.sum())

    # Floating point precision artifacts
    prec_mask = np.random.random(len(df)) < 0.08
    df.loc[prec_mask, 'total_amount'] = df.loc[prec_mask, 'total_amount'] + 0.0000001

    # Extreme outlier fares (0.5% of rows)
    outlier_mask = np.random.random(len(df)) < 0.005
    df.loc[outlier_mask, 'fare_amount'] = np.random.uniform(10000, 999999, size=outlier_mask.sum())

    # Zero-distance trips with high fares
    zero_mask = np.random.random(len(df)) < 0.02
    df.loc[zero_mask, 'trip_distance'] = 0
    df.loc[zero_mask, 'fare_amount'] = np.random.uniform(50, 500, size=zero_mask.sum())

    df.to_csv('corrupted/nyc_taxi_v1_timestamps_values.csv', index=False)
    print(f"  Saved {len(df):,} rows")

def corrupt_taxi_v2():
    """
    Corruption: Duplicates, missing values, referential integrity
    - Exact and near duplicates
    - Various NULL representations
    - Invalid location IDs
    - Inconsistent vendor IDs
    """
    print("Corrupting NYC Taxi v2: duplicates, nulls, integrity...")
    df = pd.read_csv('raw/nyc_taxi_jan2024.csv')

    # Add exact duplicates (3% of dataset)
    n_dupes = int(len(df) * 0.03)
    dupes = df.sample(n=n_dupes, random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)

    # Add near duplicates (same trip, slightly different values)
    n_near = int(len(df) * 0.02)
    near_dupes = df.sample(n=n_near, random_state=43).copy()
    near_dupes['total_amount'] = near_dupes['total_amount'] + np.random.uniform(-0.5, 0.5, size=n_near)
    near_dupes['tip_amount'] = near_dupes['tip_amount'] + np.random.uniform(-0.25, 0.25, size=n_near)
    df = pd.concat([df, near_dupes], ignore_index=True)

    # Various NULL representations - convert columns to object type first
    null_values = ['', 'NULL', 'None', 'N/A', 'NA', 'null', 'NaN', '-', '.']
    for col in ['passenger_count', 'RatecodeID', 'payment_type', 'congestion_surcharge']:
        df[col] = df[col].astype(object)
        null_mask = np.random.random(len(df)) < 0.06
        df.loc[null_mask, col] = np.random.choice(null_values, size=null_mask.sum())

    # Invalid location IDs (valid range is typically 1-265)
    invalid_loc_mask = np.random.random(len(df)) < 0.04
    df.loc[invalid_loc_mask, 'PULocationID'] = np.random.choice([0, -1, 999, 9999], size=invalid_loc_mask.sum())
    df.loc[invalid_loc_mask, 'DOLocationID'] = np.random.choice([0, -1, 888, 9999], size=invalid_loc_mask.sum())

    # Inconsistent vendor ID representations
    df['VendorID'] = df['VendorID'].astype(object)
    vendor_mask = np.random.random(len(df)) < 0.05
    vendor_map = {1: ['1', 'CMT', 'Creative Mobile', 'one'],
                  2: ['2', 'VTS', 'VeriFone', 'two']}
    df.loc[vendor_mask, 'VendorID'] = df.loc[vendor_mask, 'VendorID'].apply(
        lambda x: random.choice(vendor_map.get(int(x) if pd.notna(x) else 0, [x])) if pd.notna(x) else x
    )

    # Shuffle to mix duplicates throughout
    df = df.sample(frac=1, random_state=44).reset_index(drop=True)

    df.to_csv('corrupted/nyc_taxi_v2_duplicates_nulls.csv', index=False)
    print(f"  Saved {len(df):,} rows")

# =============================================================================
# ONLINE RETAIL CORRUPTIONS
# =============================================================================

def corrupt_retail_v1():
    """
    Corruption: Encoding issues, special characters, whitespace
    - Mixed encodings
    - Unicode problems
    - Leading/trailing whitespace
    - Embedded newlines and tabs
    """
    print("Corrupting Online Retail v1: encoding, special chars...")
    df = pd.read_csv('raw/online_retail.csv')

    # Sample to 150K for manageable size
    if len(df) > 150000:
        df = df.sample(n=150000, random_state=42)

    # Add special characters to descriptions
    special_chars = ['é', 'ñ', 'ü', 'ø', 'ß', 'æ', 'ð', '™', '©', '®', '€', '£', '¥']
    desc_mask = np.random.random(len(df)) < 0.08
    df.loc[desc_mask, 'Description'] = df.loc[desc_mask, 'Description'].apply(
        lambda x: f"{random.choice(special_chars)}{x}" if pd.notna(x) else x
    )

    # Leading/trailing whitespace
    ws_mask = np.random.random(len(df)) < 0.12
    spaces = ['  ', '   ', '\t', ' \t ']
    df.loc[ws_mask, 'Description'] = df.loc[ws_mask, 'Description'].apply(
        lambda x: f"{random.choice(spaces)}{x}{random.choice(spaces)}" if pd.notna(x) else x
    )

    # Inconsistent case in Country
    case_mask = np.random.random(len(df)) < 0.15
    df.loc[case_mask, 'Country'] = df.loc[case_mask, 'Country'].apply(
        lambda x: random.choice([str(x).upper(), str(x).lower(), str(x).title()]) if pd.notna(x) else x
    )

    # StockCode format inconsistencies (some have letters, some don't)
    stock_mask = np.random.random(len(df)) < 0.08
    df.loc[stock_mask, 'StockCode'] = df.loc[stock_mask, 'StockCode'].apply(
        lambda x: str(x).lstrip('0').lower() if pd.notna(x) else x
    )

    # Mixed quote styles in Description (can cause CSV parsing issues)
    quote_mask = np.random.random(len(df)) < 0.05
    df.loc[quote_mask, 'Description'] = df.loc[quote_mask, 'Description'].apply(
        lambda x: f'"{x}"' if pd.notna(x) and '"' not in str(x) else x
    )

    df.to_csv('corrupted/online_retail_v1_encoding.csv', index=False)
    print(f"  Saved {len(df):,} rows")

def corrupt_retail_v2():
    """
    Corruption: Business logic violations, calculation errors
    - Negative quantities for non-returns
    - Price * quantity != line total
    - Future invoice dates
    - Invalid customer IDs
    - Cancelled orders mixed with regular
    """
    print("Corrupting Online Retail v2: business logic violations...")
    df = pd.read_csv('raw/online_retail.csv')

    if len(df) > 150000:
        df = df.sample(n=150000, random_state=42)

    # Negative quantities without cancellation indicator
    neg_qty_mask = np.random.random(len(df)) < 0.06
    # Only corrupt rows that don't start with 'C' (cancellation)
    invoice_not_cancel = ~df['InvoiceNo'].astype(str).str.startswith('C')
    apply_mask = neg_qty_mask & invoice_not_cancel
    df.loc[apply_mask, 'Quantity'] = -abs(df.loc[apply_mask, 'Quantity'])

    # Zero prices (should be > 0 for valid transactions)
    zero_price_mask = np.random.random(len(df)) < 0.04
    df.loc[zero_price_mask, 'UnitPrice'] = 0

    # Extremely high prices
    high_price_mask = np.random.random(len(df)) < 0.02
    df.loc[high_price_mask, 'UnitPrice'] = np.random.uniform(10000, 999999, size=high_price_mask.sum())

    # Future invoice dates
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).astype(str)
    future_mask = np.random.random(len(df)) < 0.03
    future_dates = pd.date_range(start='2026-01-01', periods=future_mask.sum(), freq='h').strftime('%Y-%m-%d %H:%M:%S')
    df.loc[future_mask, 'InvoiceDate'] = list(future_dates[:future_mask.sum()])

    # Invalid CustomerID (should be numeric)
    df['CustomerID'] = df['CustomerID'].astype(object)
    invalid_cust_mask = np.random.random(len(df)) < 0.05
    invalid_ids = ['GUEST', 'UNKNOWN', 'TEST', '-1', 'NULL', '0']
    df.loc[invalid_cust_mask, 'CustomerID'] = np.random.choice(invalid_ids, size=invalid_cust_mask.sum())

    # Duplicate InvoiceNo with different items (should be unique per transaction)
    # This creates orphaned line items
    dupe_inv_mask = np.random.random(len(df)) < 0.03
    existing_invoices = df.loc[~dupe_inv_mask, 'InvoiceNo'].dropna().unique()
    if len(existing_invoices) > 0:
        df.loc[dupe_inv_mask, 'InvoiceNo'] = np.random.choice(existing_invoices, size=dupe_inv_mask.sum())

    df.to_csv('corrupted/online_retail_v2_logic.csv', index=False)
    print(f"  Saved {len(df):,} rows")

# =============================================================================
# CHICAGO CRIMES CORRUPTIONS
# =============================================================================

def corrupt_crimes_v1():
    """
    Corruption: Geographic data issues, coordinate problems
    - Swapped lat/long
    - Coordinates outside Chicago
    - Missing location data (systematic by type)
    - Inconsistent district/ward mappings
    """
    print("Corrupting Chicago Crimes v1: geographic issues...")
    df = pd.read_csv('raw/chicago_crimes.csv')

    # Swap lat/long (5% of rows)
    swap_mask = np.random.random(len(df)) < 0.05
    lat_temp = df.loc[swap_mask, 'latitude'].copy()
    df.loc[swap_mask, 'latitude'] = df.loc[swap_mask, 'longitude']
    df.loc[swap_mask, 'longitude'] = lat_temp

    # Coordinates outside Chicago bounds (41.6-42.1 lat, -88.0 to -87.5 long)
    out_bounds_mask = np.random.random(len(df)) < 0.04
    df.loc[out_bounds_mask, 'latitude'] = np.random.uniform(30, 50, size=out_bounds_mask.sum())
    df.loc[out_bounds_mask, 'longitude'] = np.random.uniform(-100, -70, size=out_bounds_mask.sum())

    # Zero coordinates
    zero_mask = np.random.random(len(df)) < 0.03
    df.loc[zero_mask, 'latitude'] = 0
    df.loc[zero_mask, 'longitude'] = 0

    # Systematic missing location data for certain crime types
    # Drug crimes often have redacted locations
    drug_mask = df['primary_type'].str.contains('NARCOTICS|DRUG', case=False, na=False)
    redact_mask = drug_mask & (np.random.random(len(df)) < 0.30)
    df.loc[redact_mask, 'latitude'] = np.nan
    df.loc[redact_mask, 'longitude'] = np.nan
    df.loc[redact_mask, 'block'] = 'REDACTED'

    # Inconsistent district/ward (valid districts are 1-25, wards 1-50)
    bad_district_mask = np.random.random(len(df)) < 0.04
    df.loc[bad_district_mask, 'district'] = np.random.choice([0, -1, 99, 100], size=bad_district_mask.sum())

    bad_ward_mask = np.random.random(len(df)) < 0.04
    df.loc[bad_ward_mask, 'ward'] = np.random.choice([0, -1, 99, 100], size=bad_ward_mask.sum())

    df.to_csv('corrupted/chicago_crimes_v1_geographic.csv', index=False)
    print(f"  Saved {len(df):,} rows")

def corrupt_crimes_v2():
    """
    Corruption: Temporal and categorical issues
    - Date format inconsistencies
    - Mismatched IUCR codes and descriptions
    - Boolean inconsistencies (arrest/domestic)
    - Case number duplicates with conflicts
    """
    print("Corrupting Chicago Crimes v2: temporal, categorical...")
    df = pd.read_csv('raw/chicago_crimes.csv')

    # Date format chaos
    df['date'] = pd.to_datetime(df['date']).astype(str)
    formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p', '%d-%m-%Y %H:%M',
               '%Y/%m/%d %H:%M:%S', '%m-%d-%Y %H:%M:%S']
    format_mask = np.random.random(len(df)) < 0.25
    df.loc[format_mask, 'date'] = df.loc[format_mask, 'date'].apply(
        lambda x: pd.to_datetime(x).strftime(random.choice(formats)) if pd.notna(x) and x != 'NaT' else x
    )

    # IUCR code / primary_type mismatch
    # Swap some codes between different crime types
    crime_types = df['primary_type'].dropna().unique()
    mismatch_mask = np.random.random(len(df)) < 0.06
    df.loc[mismatch_mask, 'primary_type'] = np.random.choice(crime_types, size=mismatch_mask.sum())

    # Boolean inconsistencies (text vs bool vs int)
    bool_values = ['True', 'False', 'true', 'false', 'TRUE', 'FALSE',
                   '1', '0', 'Yes', 'No', 'Y', 'N']
    df['arrest'] = df['arrest'].astype(object)
    arrest_mask = np.random.random(len(df)) < 0.15
    df.loc[arrest_mask, 'arrest'] = np.random.choice(bool_values, size=arrest_mask.sum())

    df['domestic'] = df['domestic'].astype(object)
    domestic_mask = np.random.random(len(df)) < 0.15
    df.loc[domestic_mask, 'domestic'] = np.random.choice(bool_values, size=domestic_mask.sum())

    # Case number duplicates with conflicting data
    n_dupe_cases = int(len(df) * 0.02)
    dupe_idx = df.sample(n=n_dupe_cases, random_state=42).index
    dupes = df.loc[dupe_idx].copy()
    # Keep same case number but change other fields
    dupes['arrest'] = dupes['arrest'].apply(lambda x: 'False' if str(x).lower() in ['true', '1', 'yes', 'y'] else 'True')
    dupes['district'] = pd.to_numeric(dupes['district'], errors='coerce').fillna(0).astype(int) + 1
    df = pd.concat([df, dupes], ignore_index=True)

    # Year column doesn't match date
    year_mismatch_mask = np.random.random(len(df)) < 0.05
    df.loc[year_mismatch_mask, 'year'] = df.loc[year_mismatch_mask, 'year'].apply(
        lambda x: x - random.randint(1, 5) if pd.notna(x) else x
    )

    df = df.sample(frac=1, random_state=45).reset_index(drop=True)
    df.to_csv('corrupted/chicago_crimes_v2_temporal.csv', index=False)
    print(f"  Saved {len(df):,} rows")

# =============================================================================
# AIR QUALITY CORRUPTIONS
# =============================================================================

def corrupt_airquality_v1():
    """
    Corruption: Sensor data issues, unit problems
    - Mixed units (µg/m³ vs ppm)
    - Out of range AQI values
    - Negative concentrations
    - Missing required methodology fields
    """
    print("Corrupting Air Quality v1: sensor/unit issues...")
    df = pd.read_csv('raw/air_quality_pm25.csv')

    # Unit inconsistencies
    unit_mask = np.random.random(len(df)) < 0.12
    units = ['Micrograms/cubic meter (LC)', 'µg/m³', 'ug/m3', 'ppm', 'ppb', 'mg/m3']
    df.loc[unit_mask, 'Units of Measure'] = np.random.choice(units, size=unit_mask.sum())

    # Negative concentration values (physically impossible)
    neg_mask = np.random.random(len(df)) < 0.04
    df.loc[neg_mask, 'Arithmetic Mean'] = -abs(df.loc[neg_mask, 'Arithmetic Mean'])

    # Out of range AQI (valid 0-500)
    aqi_mask = np.random.random(len(df)) < 0.03
    df.loc[aqi_mask, 'AQI'] = np.random.choice([-10, 600, 999, 1000], size=aqi_mask.sum())

    # Extreme outliers in 1st Max Value
    outlier_mask = np.random.random(len(df)) < 0.02
    df.loc[outlier_mask, '1st Max Value'] = np.random.uniform(1000, 10000, size=outlier_mask.sum())

    # Observation percent > 100 (impossible)
    df['Observation Percent'] = df['Observation Percent'].astype(float)
    obs_pct_mask = np.random.random(len(df)) < 0.03
    df.loc[obs_pct_mask, 'Observation Percent'] = np.random.uniform(100.1, 200, size=obs_pct_mask.sum())

    # Method Code / Method Name mismatch
    method_mask = np.random.random(len(df)) < 0.05
    df.loc[method_mask, 'Method Name'] = 'UNKNOWN METHOD'

    # Invalid state codes (should be 2-digit numeric)
    df['State Code'] = df['State Code'].astype(object)
    state_mask = np.random.random(len(df)) < 0.04
    df.loc[state_mask, 'State Code'] = np.random.choice(['XX', '99', '-1', 'NA'], size=state_mask.sum())

    df.to_csv('corrupted/air_quality_v1_sensors.csv', index=False)
    print(f"  Saved {len(df):,} rows")

def corrupt_airquality_v2():
    """
    Corruption: Temporal and geographic issues
    - Date gaps and overlaps
    - Coordinate precision loss
    - State/County code mismatches
    - Duplicate monitoring records
    """
    print("Corrupting Air Quality v2: temporal, geographic...")
    df = pd.read_csv('raw/air_quality_pm25.csv')

    # Date format inconsistencies
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
    date_mask = np.random.random(len(df)) < 0.20
    df.loc[date_mask, 'Date Local'] = df.loc[date_mask, 'Date Local'].apply(
        lambda x: pd.to_datetime(x).strftime(random.choice(formats)) if pd.notna(x) else x
    )

    # Future dates
    future_mask = np.random.random(len(df)) < 0.02
    future_dates = pd.date_range(start='2026-01-01', periods=future_mask.sum(), freq='d').strftime('%Y-%m-%d')
    df.loc[future_mask, 'Date Local'] = list(future_dates[:future_mask.sum()])

    # Coordinate precision loss (rounded too much)
    coord_mask = np.random.random(len(df)) < 0.08
    df.loc[coord_mask, 'Latitude'] = df.loc[coord_mask, 'Latitude'].apply(
        lambda x: round(x, 1) if pd.notna(x) else x
    )
    df.loc[coord_mask, 'Longitude'] = df.loc[coord_mask, 'Longitude'].apply(
        lambda x: round(x, 1) if pd.notna(x) else x
    )

    # State name / State code mismatch
    states = df['State Name'].dropna().unique()
    mismatch_mask = np.random.random(len(df)) < 0.05
    df.loc[mismatch_mask, 'State Name'] = np.random.choice(states, size=mismatch_mask.sum())

    # Duplicate site readings (same location, same day)
    n_dupes = int(len(df) * 0.03)
    dupes = df.sample(n=n_dupes, random_state=42).copy()
    # Slightly different readings
    dupes['Arithmetic Mean'] = dupes['Arithmetic Mean'] * np.random.uniform(0.9, 1.1, size=n_dupes)
    dupes['AQI'] = dupes['AQI'] + np.random.randint(-5, 5, size=n_dupes)
    df = pd.concat([df, dupes], ignore_index=True)

    # CBSA Name inconsistencies (same area, different spellings)
    cbsa_mask = np.random.random(len(df)) < 0.06
    df.loc[cbsa_mask, 'CBSA Name'] = df.loc[cbsa_mask, 'CBSA Name'].apply(
        lambda x: str(x).upper() if pd.notna(x) and random.random() > 0.5 else (
            str(x).replace('-', ' ').replace(',', '') if pd.notna(x) else x
        )
    )

    df = df.sample(frac=1, random_state=46).reset_index(drop=True)
    df.to_csv('corrupted/air_quality_v2_temporal.csv', index=False)
    print(f"  Saved {len(df):,} rows")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA QUALITY POC - Corrupting Real Datasets")
    print("=" * 60)
    print()

    # Check all raw files exist
    raw_files = ['raw/nyc_taxi_jan2024.csv', 'raw/online_retail.csv',
                 'raw/chicago_crimes.csv', 'raw/air_quality_pm25.csv']

    missing = [f for f in raw_files if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing raw files: {missing}")
        print("Run the download script first.")
        exit(1)

    # NYC Taxi
    corrupt_taxi_v1()
    corrupt_taxi_v2()
    print()

    # Online Retail
    corrupt_retail_v1()
    corrupt_retail_v2()
    print()

    # Chicago Crimes
    corrupt_crimes_v1()
    corrupt_crimes_v2()
    print()

    # Air Quality
    corrupt_airquality_v1()
    corrupt_airquality_v2()
    print()

    print("=" * 60)
    print("CORRUPTION COMPLETE")
    print("=" * 60)
    print()
    print("Corrupted datasets in: ./corrupted/")
    print()
    print("NYC Taxi:")
    print("  - v1_timestamps_values: Mixed timestamps, impossible values")
    print("  - v2_duplicates_nulls: Duplicates, NULL representations")
    print()
    print("Online Retail:")
    print("  - v1_encoding: Special chars, whitespace, case issues")
    print("  - v2_logic: Business logic violations")
    print()
    print("Chicago Crimes:")
    print("  - v1_geographic: Coordinate issues, invalid locations")
    print("  - v2_temporal: Date formats, categorical mismatches")
    print()
    print("Air Quality:")
    print("  - v1_sensors: Unit issues, impossible values")
    print("  - v2_temporal: Date issues, coordinate precision, duplicates")
