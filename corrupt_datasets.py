#!/usr/bin/env python3
"""
Corrupt real-world datasets with realistic data quality issues.

Usage:
    python corrupt_datasets.py [options]

Options:
    --randomize         Randomly select 1-8 corruptions per dataset (default: all)
    --scale FLOAT       Scale corruption percentages (0.5 = half rate, 2.0 = double)
    --placebo           Also generate clean "placebo" versions for comparison
    --include-rare      Include rare corruptions (0.5-1% rate)
    --include-timeboxed Include timeboxed corruptions (contiguous blocks)
    --seed INT          Random seed for reproducibility (default: 42)
    --output-dir DIR    Output directory (default: corrupted/)
    --help              Show this help message

Examples:
    # Standard corruption (all issues, default percentages)
    python corrupt_datasets.py

    # Randomized subset with placebo files
    python corrupt_datasets.py --randomize --placebo

    # Half the usual corruption rate with rare issues
    python corrupt_datasets.py --scale 0.5 --include-rare

    # Full chaos mode
    python corrupt_datasets.py --randomize --include-rare --include-timeboxed --placebo
"""

import pandas as pd
import numpy as np
import random
import os
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any
from enum import Enum


class CorruptionType(Enum):
    STANDARD = "standard"
    RARE = "rare"
    TIMEBOXED = "timeboxed"


@dataclass
class Corruption:
    """Definition of a single corruption to apply."""
    name: str
    description: str
    base_percentage: float
    apply_fn: Callable[[pd.DataFrame, float, bool], pd.DataFrame]
    corruption_type: CorruptionType = CorruptionType.STANDARD
    columns_affected: List[str] = field(default_factory=list)


def apply_random_mask(df: pd.DataFrame, percentage: float, timeboxed: bool = False) -> np.ndarray:
    """
    Generate a boolean mask for rows to corrupt.

    Args:
        df: DataFrame to corrupt
        percentage: Percentage of rows to affect (0-100)
        timeboxed: If True, affected rows are contiguous instead of random
    """
    n_rows = len(df)
    n_affected = int(n_rows * (percentage / 100))

    if timeboxed and n_affected > 0:
        # Pick a random starting point and make a contiguous block
        max_start = max(0, n_rows - n_affected)
        start_idx = random.randint(0, max_start)
        mask = np.zeros(n_rows, dtype=bool)
        mask[start_idx:start_idx + n_affected] = True
    else:
        # Random scatter throughout dataset
        mask = np.random.random(n_rows) < (percentage / 100)

    return mask


# =============================================================================
# CORRUPTION FUNCTIONS
# =============================================================================

def corrupt_timestamp_format(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'tpep_pickup_datetime') -> pd.DataFrame:
    """Mix timestamp formats."""
    formats = ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M', '%d-%m-%Y %H:%M:%S',
               '%Y/%m/%d %H:%M:%S', '%Y-%m-%dT%H:%M:%SZ']
    mask = apply_random_mask(df, pct, timeboxed)
    df[col] = df[col].astype(str)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: pd.to_datetime(x).strftime(random.choice(formats)) if pd.notna(x) and x != 'NaT' else x
    )
    return df


def corrupt_future_timestamps(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'tpep_pickup_datetime') -> pd.DataFrame:
    """Set timestamps to future dates (2026)."""
    mask = apply_random_mask(df, pct, timeboxed)
    future_dates = [
        datetime(2026, random.randint(1, 12), random.randint(1, 28),
                 random.randint(0, 23), random.randint(0, 59)).strftime('%Y-%m-%d %H:%M:%S')
        for _ in range(mask.sum())
    ]
    df[col] = df[col].astype(str)
    df.loc[mask, col] = future_dates
    return df


def corrupt_negative_values(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'fare_amount') -> pd.DataFrame:
    """Multiply values by -1."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = -abs(df.loc[mask, col])
    return df


def corrupt_impossible_values(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'passenger_count',
                               values: List[Any] = None) -> pd.DataFrame:
    """Replace with impossible/invalid values."""
    if values is None:
        values = [-1, -2, 15, 99, 127]
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = np.random.choice(values, size=mask.sum())
    return df


def corrupt_precision_artifacts(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'total_amount') -> pd.DataFrame:
    """Add floating point precision artifacts."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col] + 0.0000001
    return df


def corrupt_extreme_outliers(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'fare_amount',
                              low: float = 10000, high: float = 999999) -> pd.DataFrame:
    """Set values to extreme outliers."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = np.random.uniform(low, high, size=mask.sum())
    return df


def corrupt_zero_with_conflict(df: pd.DataFrame, pct: float, timeboxed: bool,
                                zero_col: str = 'trip_distance', conflict_col: str = 'fare_amount') -> pd.DataFrame:
    """Set one column to zero while another has high value (logic violation)."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, zero_col] = 0
    df.loc[mask, conflict_col] = np.random.uniform(50, 500, size=mask.sum())
    return df


def corrupt_temporal_logic(df: pd.DataFrame, pct: float, timeboxed: bool,
                           start_col: str = 'tpep_pickup_datetime', end_col: str = 'tpep_dropoff_datetime') -> pd.DataFrame:
    """Make end time before start time."""
    mask = apply_random_mask(df, pct, timeboxed)
    df[end_col] = df[end_col].astype(str)
    df.loc[mask, end_col] = df.loc[mask, start_col].apply(
        lambda x: (pd.to_datetime(x) - timedelta(hours=random.randint(1, 5))).strftime('%Y-%m-%d %H:%M:%S')
        if pd.notna(x) and x != 'NaT' else x
    )
    return df


def corrupt_invalid_ids(df: pd.DataFrame, pct: float, timeboxed: bool, cols: List[str] = None,
                         invalid_values: List[Any] = None) -> pd.DataFrame:
    """Set ID columns to invalid values."""
    if cols is None:
        cols = ['PULocationID', 'DOLocationID']
    if invalid_values is None:
        invalid_values = [0, 888, 999, 9999]
    mask = apply_random_mask(df, pct, timeboxed)
    for col in cols:
        df.loc[mask, col] = np.random.choice(invalid_values, size=mask.sum())
    return df


def corrupt_special_chars(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'Description') -> pd.DataFrame:
    """Prepend special characters."""
    chars = ['é', 'ñ', 'ü', 'ø', 'ß', 'æ', '™', '©', '®', '€', '£']
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: f"{random.choice(chars)}{x}" if pd.notna(x) else x
    )
    return df


def corrupt_whitespace(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'Description') -> pd.DataFrame:
    """Add leading/trailing whitespace."""
    spaces = ['  ', '   ', '\t', ' \t ']
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: f"{random.choice(spaces)}{x}{random.choice(spaces)}" if pd.notna(x) else x
    )
    return df


def corrupt_case_inconsistency(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'Country') -> pd.DataFrame:
    """Randomize case."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: random.choice([str(x).upper(), str(x).lower(), str(x).title()]) if pd.notna(x) else x
    )
    return df


def corrupt_embedded_quotes(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'Description') -> pd.DataFrame:
    """Wrap values in extra quotes."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: f'"{x}"' if pd.notna(x) and '"' not in str(x) else x
    )
    return df


def corrupt_null_strings(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'Description') -> pd.DataFrame:
    """Replace with various null string representations."""
    null_values = ['', 'NULL', 'N/A', 'nan', 'None', '-']
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = np.random.choice(null_values, size=mask.sum())
    return df


def corrupt_id_format(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'StockCode') -> pd.DataFrame:
    """Strip leading zeros and lowercase."""
    mask = apply_random_mask(df, pct, timeboxed)
    df[col] = df[col].astype(str)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: str(x).lstrip('0').lower() if pd.notna(x) else x
    )
    return df


def corrupt_invoice_prefix(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'InvoiceNo') -> pd.DataFrame:
    """Add various prefixes to invoice numbers."""
    prefixes = ['INV_', '#', 'ORDER-', 'ORD']
    mask = apply_random_mask(df, pct, timeboxed)
    df[col] = df[col].astype(str)
    df.loc[mask, col] = df.loc[mask, col].apply(lambda x: f"{random.choice(prefixes)}{x}")
    return df


def corrupt_swap_coordinates(df: pd.DataFrame, pct: float, timeboxed: bool,
                              lat_col: str = 'latitude', lon_col: str = 'longitude') -> pd.DataFrame:
    """Swap latitude and longitude."""
    mask = apply_random_mask(df, pct, timeboxed)
    lat_temp = df.loc[mask, lat_col].copy()
    df.loc[mask, lat_col] = df.loc[mask, lon_col]
    df.loc[mask, lon_col] = lat_temp
    return df


def corrupt_out_of_bounds_coords(df: pd.DataFrame, pct: float, timeboxed: bool,
                                  lat_col: str = 'latitude', lon_col: str = 'longitude') -> pd.DataFrame:
    """Set coordinates to out-of-bounds values."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, lat_col] = np.random.uniform(30, 50, size=mask.sum())
    df.loc[mask, lon_col] = np.random.uniform(-100, -70, size=mask.sum())
    return df


def corrupt_zero_coordinates(df: pd.DataFrame, pct: float, timeboxed: bool,
                              lat_col: str = 'latitude', lon_col: str = 'longitude') -> pd.DataFrame:
    """Set coordinates to (0, 0) - null island."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, lat_col] = 0
    df.loc[mask, lon_col] = 0
    return df


def corrupt_systematic_redaction(df: pd.DataFrame, pct: float, timeboxed: bool,
                                  filter_col: str = 'primary_type', filter_pattern: str = 'NARCOTICS|DRUG',
                                  redact_cols: dict = None) -> pd.DataFrame:
    """Systematically redact data for specific categories."""
    if redact_cols is None:
        redact_cols = {'latitude': np.nan, 'longitude': np.nan, 'block': 'REDACTED'}

    type_mask = df[filter_col].str.contains(filter_pattern, case=False, na=False)
    random_mask = apply_random_mask(df, pct, timeboxed)
    combined_mask = type_mask & random_mask

    for col, value in redact_cols.items():
        df.loc[combined_mask, col] = value
    return df


def corrupt_location_suffix(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'block') -> pd.DataFrame:
    """Add suffixes to location strings."""
    suffixes = [' (APPROX)', ' - VERIFIED', ' [REDACTED]', ' *', ' ???']
    mask = apply_random_mask(df, pct, timeboxed)
    df[col] = df[col].astype(str)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: f"{x}{random.choice(suffixes)}" if pd.notna(x) else x
    )
    return df


def corrupt_precision_loss(df: pd.DataFrame, pct: float, timeboxed: bool,
                            cols: List[str] = None, round_to: int = 1000) -> pd.DataFrame:
    """Round coordinates/values to lose precision."""
    if cols is None:
        cols = ['x_coordinate', 'y_coordinate']
    mask = apply_random_mask(df, pct, timeboxed)
    for col in cols:
        df.loc[mask, col] = (df.loc[mask, col] / round_to).round() * round_to
    return df


def corrupt_unit_conversion(df: pd.DataFrame, pct: float, timeboxed: bool,
                             unit_col: str = 'Units of Measure', value_cols: List[str] = None) -> pd.DataFrame:
    """Change units WITH proper value conversion (recoverable)."""
    if value_cols is None:
        value_cols = ['Arithmetic Mean', '1st Max Value']

    unit_names = ['µg/m³', 'ug/m3', 'Micrograms/cubic meter', 'mg/m3', 'ng/m3']
    unit_factors = [1, 1, 1, 0.001, 1000]

    mask = apply_random_mask(df, pct, timeboxed)
    n_affected = mask.sum()

    unit_indices = np.random.randint(0, len(unit_names), size=n_affected)
    chosen_units = [unit_names[i] for i in unit_indices]
    chosen_factors = np.array([unit_factors[i] for i in unit_indices])

    df.loc[mask, unit_col] = chosen_units
    for col in value_cols:
        df.loc[mask, col] = df.loc[mask, col].values * chosen_factors

    return df


def corrupt_boolean_format(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'arrest') -> pd.DataFrame:
    """Use various boolean representations."""
    bool_values = ['True', 'False', 'true', 'false', 'YES', 'NO', '1', '0', 'Y', 'N']
    mask = apply_random_mask(df, pct, timeboxed)
    df[col] = df[col].astype(object)
    df.loc[mask, col] = np.random.choice(bool_values, size=mask.sum())
    return df


def corrupt_code_mismatch(df: pd.DataFrame, pct: float, timeboxed: bool,
                           col: str = 'primary_type') -> pd.DataFrame:
    """Swap values to create code/description mismatches."""
    unique_values = df[col].dropna().unique()
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = np.random.choice(unique_values, size=mask.sum())
    return df


def corrupt_duplicate_with_conflict(df: pd.DataFrame, pct: float, timeboxed: bool,
                                     key_col: str = 'case_number', conflict_cols: dict = None) -> pd.DataFrame:
    """Create duplicate keys with conflicting values."""
    if conflict_cols is None:
        conflict_cols = {'arrest': lambda x: 'False' if str(x).lower() in ['true', '1', 'yes', 'y'] else 'True',
                         'district': lambda x: int(pd.to_numeric(x, errors='coerce') or 0) + 1}

    n_dupes = int(len(df) * (pct / 100))
    dupe_idx = df.sample(n=n_dupes, random_state=42).index
    dupes = df.loc[dupe_idx].copy()

    for col, transform in conflict_cols.items():
        dupes[col] = dupes[col].apply(transform)

    df = pd.concat([df, dupes], ignore_index=True)
    return df


def corrupt_derived_field_mismatch(df: pd.DataFrame, pct: float, timeboxed: bool,
                                    col: str = 'year', offset_range: tuple = (1, 5)) -> pd.DataFrame:
    """Make derived field not match source."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: int(x) - random.randint(*offset_range) if pd.notna(x) else x
    )
    return df


def corrupt_invalid_codes(df: pd.DataFrame, pct: float, timeboxed: bool,
                           col: str = 'fbi_code', invalid_values: List[str] = None) -> pd.DataFrame:
    """Set codes to invalid values."""
    if invalid_values is None:
        invalid_values = ['XX', '00', 'UNKNOWN', '']
    mask = apply_random_mask(df, pct, timeboxed)
    df[col] = df[col].astype(object)
    df.loc[mask, col] = np.random.choice(invalid_values, size=mask.sum())
    return df


def corrupt_exact_duplicates(df: pd.DataFrame, pct: float, timeboxed: bool) -> pd.DataFrame:
    """Add exact row duplicates."""
    n_dupes = int(len(df) * (pct / 100))
    dupes = df.sample(n=n_dupes, random_state=42)
    df = pd.concat([df, dupes], ignore_index=True)
    return df


def corrupt_near_duplicates(df: pd.DataFrame, pct: float, timeboxed: bool,
                             vary_cols: dict = None) -> pd.DataFrame:
    """Add near-duplicate rows with slight variations."""
    if vary_cols is None:
        vary_cols = {'total_amount': (-0.5, 0.5), 'tip_amount': (-0.25, 0.25)}

    n_dupes = int(len(df) * (pct / 100))
    dupes = df.sample(n=n_dupes, random_state=43).copy()

    for col, (low, high) in vary_cols.items():
        if col in dupes.columns:
            dupes[col] = dupes[col] + np.random.uniform(low, high, size=n_dupes)

    df = pd.concat([df, dupes], ignore_index=True)
    return df


def corrupt_null_variations(df: pd.DataFrame, pct: float, timeboxed: bool,
                             cols: List[str] = None) -> pd.DataFrame:
    """Apply various null representations to multiple columns."""
    null_values = ['', 'NULL', 'None', 'N/A', 'null', 'NaN', '-', '.']
    if cols is None:
        cols = ['passenger_count', 'RatecodeID', 'payment_type']

    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(object)
            mask = apply_random_mask(df, pct, timeboxed)
            df.loc[mask, col] = np.random.choice(null_values, size=mask.sum())

    return df


def corrupt_vendor_format(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'VendorID') -> pd.DataFrame:
    """Mix numeric IDs with text representations."""
    vendor_map = {1: ['1', 'CMT', 'Creative Mobile', 'one'],
                  2: ['2', 'VTS', 'VeriFone', 'two']}
    mask = apply_random_mask(df, pct, timeboxed)
    df[col] = df[col].astype(object)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: random.choice(vendor_map.get(int(x) if pd.notna(x) and str(x).isdigit() else 0, [x]))
        if pd.notna(x) else x
    )
    return df


def corrupt_payment_type_text(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'payment_type') -> pd.DataFrame:
    """Mix numeric codes with text descriptions."""
    pay_map = {1: 'Credit', 2: 'Cash', 3: 'No charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided'}
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: pay_map.get(int(x) if pd.notna(x) and str(x).isdigit() else 0, x)
    )
    return df


def corrupt_country_codes(df: pd.DataFrame, pct: float, timeboxed: bool, col: str = 'Country') -> pd.DataFrame:
    """Mix country names with ISO codes."""
    country_map = {
        'United Kingdom': ['UK', 'GB', 'GBR', 'Britain'],
        'France': ['FR', 'FRA', 'FRANCE'],
        'Germany': ['DE', 'DEU', 'GERMANY'],
        'Spain': ['ES', 'ESP', 'SPAIN'],
        'Netherlands': ['NL', 'NLD', 'NETHERLANDS']
    }
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, col] = df.loc[mask, col].apply(
        lambda x: random.choice(country_map.get(x, [x])) if pd.notna(x) else x
    )
    return df


def corrupt_math_errors(df: pd.DataFrame, pct: float, timeboxed: bool,
                         qty_col: str = 'Quantity', price_col: str = 'UnitPrice', total_col: str = 'LineTotal') -> pd.DataFrame:
    """Add calculated column with errors."""
    df[total_col] = df[qty_col] * df[price_col]
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, total_col] = df.loc[mask, total_col] + np.random.uniform(-10, 10, size=mask.sum())
    return df


def corrupt_coordinate_precision(df: pd.DataFrame, pct: float, timeboxed: bool,
                                  lat_col: str = 'Latitude', lon_col: str = 'Longitude', decimals: int = 1) -> pd.DataFrame:
    """Round coordinates to lose precision."""
    mask = apply_random_mask(df, pct, timeboxed)
    df.loc[mask, lat_col] = df.loc[mask, lat_col].apply(lambda x: round(x, decimals) if pd.notna(x) else x)
    df.loc[mask, lon_col] = df.loc[mask, lon_col].apply(lambda x: round(x, decimals) if pd.notna(x) else x)
    return df


# =============================================================================
# DATASET CORRUPTION DEFINITIONS
# =============================================================================

def get_taxi_v1_corruptions():
    """Define corruptions for NYC Taxi V1 (large dataset)."""
    return [
        # Standard corruptions
        Corruption("timestamp_format", "Mixed datetime formats", 25,
                   lambda df, pct, tb: corrupt_timestamp_format(df, pct, tb, 'tpep_pickup_datetime')),
        Corruption("future_timestamps", "Dates in 2026", 3,
                   lambda df, pct, tb: corrupt_future_timestamps(df, pct, tb, 'tpep_pickup_datetime')),
        Corruption("negative_fares", "Negative fare amounts", 5,
                   lambda df, pct, tb: corrupt_negative_values(df, pct, tb, 'fare_amount')),
        Corruption("negative_distances", "Negative trip distances", 4,
                   lambda df, pct, tb: corrupt_negative_values(df, pct, tb, 'trip_distance')),
        Corruption("impossible_passengers", "Invalid passenger counts", 3,
                   lambda df, pct, tb: corrupt_impossible_values(df, pct, tb, 'passenger_count', [-1, -2, 15, 99, 127])),
        Corruption("precision_artifacts", "Floating point artifacts", 8,
                   lambda df, pct, tb: corrupt_precision_artifacts(df, pct, tb, 'total_amount')),
        Corruption("zero_distance_high_fare", "Zero distance with high fare", 2,
                   lambda df, pct, tb: corrupt_zero_with_conflict(df, pct, tb, 'trip_distance', 'fare_amount')),
        Corruption("invalid_location_ids", "Invalid pickup/dropoff IDs", 2,
                   lambda df, pct, tb: corrupt_invalid_ids(df, pct, tb, ['PULocationID', 'DOLocationID'])),

        # Rare corruptions
        Corruption("extreme_outlier_fares", "Fares $10K-$1M", 0.5,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, tb, 'fare_amount'),
                   corruption_type=CorruptionType.RARE),
        Corruption("dropoff_before_pickup", "Time travel (dropoff < pickup)", 1.5,
                   lambda df, pct, tb: corrupt_temporal_logic(df, pct, tb),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed corruptions
        Corruption("holiday_surge_format", "Holiday period format change", 5,
                   lambda df, pct, tb: corrupt_timestamp_format(df, pct, True, 'tpep_pickup_datetime'),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


def get_retail_v1_corruptions():
    """Define corruptions for Online Retail V1."""
    return [
        Corruption("special_chars", "Special characters in descriptions", 10,
                   lambda df, pct, tb: corrupt_special_chars(df, pct, tb, 'Description')),
        Corruption("whitespace", "Leading/trailing whitespace", 15,
                   lambda df, pct, tb: corrupt_whitespace(df, pct, tb, 'Description')),
        Corruption("case_inconsistency", "Country case variations", 20,
                   lambda df, pct, tb: corrupt_case_inconsistency(df, pct, tb, 'Country')),
        Corruption("stockcode_format", "StockCode format issues", 12,
                   lambda df, pct, tb: corrupt_id_format(df, pct, tb, 'StockCode')),
        Corruption("embedded_quotes", "Extra quotes in descriptions", 6,
                   lambda df, pct, tb: corrupt_embedded_quotes(df, pct, tb, 'Description')),
        Corruption("null_descriptions", "Various NULL representations", 8,
                   lambda df, pct, tb: corrupt_null_strings(df, pct, tb, 'Description')),
        Corruption("negative_prices", "Negative unit prices", 5,
                   lambda df, pct, tb: corrupt_negative_values(df, pct, tb, 'UnitPrice')),
        Corruption("invoice_format", "Invoice number prefixes", 4,
                   lambda df, pct, tb: corrupt_invoice_prefix(df, pct, tb, 'InvoiceNo')),

        # Rare
        Corruption("extreme_quantities", "Extreme quantity values", 0.5,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, tb, 'Quantity', -10000, 10000),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed - simulating a bad data import batch
        Corruption("bad_batch_encoding", "Batch with encoding issues", 8,
                   lambda df, pct, tb: corrupt_special_chars(df, pct, True, 'Description'),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


def get_crimes_v1_corruptions():
    """Define corruptions for Chicago Crimes V1."""
    return [
        Corruption("swapped_coords", "Swapped lat/long", 6,
                   lambda df, pct, tb: corrupt_swap_coordinates(df, pct, tb)),
        Corruption("out_of_bounds", "Coordinates outside Chicago", 5,
                   lambda df, pct, tb: corrupt_out_of_bounds_coords(df, pct, tb)),
        Corruption("zero_coords", "Null island (0,0)", 4,
                   lambda df, pct, tb: corrupt_zero_coordinates(df, pct, tb)),
        Corruption("drug_redaction", "Drug crime location redaction", 35,
                   lambda df, pct, tb: corrupt_systematic_redaction(df, pct, tb)),
        Corruption("invalid_districts", "Invalid district codes", 5,
                   lambda df, pct, tb: corrupt_impossible_values(df, pct, tb, 'district', [0, -1, 99, 100])),
        Corruption("invalid_wards", "Invalid ward codes", 5,
                   lambda df, pct, tb: corrupt_impossible_values(df, pct, tb, 'ward', [0, -1, 99, 100])),
        Corruption("location_suffixes", "Location string corruption", 8,
                   lambda df, pct, tb: corrupt_location_suffix(df, pct, tb, 'block')),
        Corruption("xy_precision_loss", "X/Y coordinate rounding", 10,
                   lambda df, pct, tb: corrupt_precision_loss(df, pct, tb)),

        # Rare
        Corruption("completely_wrong_coords", "Coords in wrong country", 0.5,
                   lambda df, pct, tb: corrupt_out_of_bounds_coords(df, pct, tb),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed - simulating GPS outage
        Corruption("gps_outage", "Period of zero coordinates", 3,
                   lambda df, pct, tb: corrupt_zero_coordinates(df, pct, True),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


def get_airquality_v1_corruptions():
    """Define corruptions for Air Quality V1."""
    return [
        Corruption("unit_conversion", "Unit changes with value conversion", 15,
                   lambda df, pct, tb: corrupt_unit_conversion(df, pct, tb)),
        Corruption("negative_concentrations", "Negative readings", 5,
                   lambda df, pct, tb: corrupt_negative_values(df, pct, tb, 'Arithmetic Mean')),
        Corruption("invalid_aqi", "AQI outside 0-500 range", 4,
                   lambda df, pct, tb: corrupt_impossible_values(df, pct, tb, 'AQI', [-10, 600, 999, 1000])),
        Corruption("extreme_max", "Extreme 1st Max Values", 3,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, tb, '1st Max Value', 1000, 10000)),
        Corruption("obs_percent_over_100", "Observation % > 100", 4,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, tb, 'Observation Percent', 100.1, 200)),
        Corruption("invalid_state_codes", "Invalid state codes", 5,
                   lambda df, pct, tb: corrupt_invalid_codes(df, pct, tb, 'State Code', ['XX', '99', '-1', 'NA'])),
        Corruption("method_mismatch", "Unknown method names", 6,
                   lambda df, pct, tb: corrupt_null_strings(df, pct, tb, 'Method Name')),
        Corruption("site_num_format", "Site number format issues", 8,
                   lambda df, pct, tb: corrupt_id_format(df, pct, tb, 'Site Num')),

        # Rare - sensor malfunction
        Corruption("sensor_spike", "Extreme sensor spike", 0.5,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, tb, 'Arithmetic Mean', 500, 2000),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed - calibration period
        Corruption("calibration_period", "Sensor calibration errors", 5,
                   lambda df, pct, tb: corrupt_negative_values(df, pct, True, 'Arithmetic Mean'),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


def get_crimes_v2_corruptions():
    """Define corruptions for Chicago Crimes V2."""
    return [
        Corruption("date_format_chaos", "Mixed date formats", 30,
                   lambda df, pct, tb: corrupt_timestamp_format(df, pct, tb, 'date')),
        Corruption("crime_type_mismatch", "Type doesn't match IUCR", 8,
                   lambda df, pct, tb: corrupt_code_mismatch(df, pct, tb, 'primary_type')),
        Corruption("arrest_boolean_chaos", "Mixed boolean formats (arrest)", 20,
                   lambda df, pct, tb: corrupt_boolean_format(df, pct, tb, 'arrest')),
        Corruption("domestic_boolean_chaos", "Mixed boolean formats (domestic)", 20,
                   lambda df, pct, tb: corrupt_boolean_format(df, pct, tb, 'domestic')),
        Corruption("duplicate_case_numbers", "Duplicate keys with conflicts", 3,
                   lambda df, pct, tb: corrupt_duplicate_with_conflict(df, pct, tb)),
        Corruption("year_mismatch", "Year doesn't match date", 6,
                   lambda df, pct, tb: corrupt_derived_field_mismatch(df, pct, tb, 'year')),
        Corruption("fbi_code_corruption", "Invalid FBI codes", 5,
                   lambda df, pct, tb: corrupt_invalid_codes(df, pct, tb, 'fbi_code')),

        # Rare
        Corruption("updated_before_date", "Update timestamp before crime", 1,
                   lambda df, pct, tb: corrupt_temporal_logic(df, pct, tb, 'date', 'updated_on'),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed - system migration
        Corruption("system_migration", "Period with different format", 10,
                   lambda df, pct, tb: corrupt_timestamp_format(df, pct, True, 'date'),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


def get_taxi_v2_corruptions():
    """Define corruptions for NYC Taxi V2."""
    return [
        Corruption("exact_duplicates", "Complete row duplicates", 5,
                   lambda df, pct, tb: corrupt_exact_duplicates(df, pct, tb)),
        Corruption("near_duplicates", "Near-duplicate trips", 4,
                   lambda df, pct, tb: corrupt_near_duplicates(df, pct, tb)),
        Corruption("null_variations", "Various NULL strings", 8,
                   lambda df, pct, tb: corrupt_null_variations(df, pct, tb, ['passenger_count', 'RatecodeID', 'payment_type', 'congestion_surcharge'])),
        Corruption("invalid_locations", "Invalid location IDs", 6,
                   lambda df, pct, tb: corrupt_invalid_ids(df, pct, tb, ['PULocationID', 'DOLocationID'], [0, -1, 999, 9999])),
        Corruption("vendor_format", "Mixed vendor ID formats", 7,
                   lambda df, pct, tb: corrupt_vendor_format(df, pct, tb)),
        Corruption("payment_type_text", "Mixed payment type formats", 6,
                   lambda df, pct, tb: corrupt_payment_type_text(df, pct, tb)),
        Corruption("ratecode_corruption", "Invalid rate codes", 5,
                   lambda df, pct, tb: corrupt_impossible_values(df, pct, tb, 'RatecodeID', [0, 7, 99, -1])),
        Corruption("store_fwd_flag_chaos", "Mixed boolean flag formats", 10,
                   lambda df, pct, tb: corrupt_boolean_format(df, pct, tb, 'store_and_fwd_flag')),

        # Rare
        Corruption("ghost_trips", "Zero everything", 0.5,
                   lambda df, pct, tb: corrupt_zero_with_conflict(df, pct, tb, 'trip_distance', 'total_amount'),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed - meter recalibration
        Corruption("meter_recalibration", "Period of duplicate entries", 3,
                   lambda df, pct, tb: corrupt_exact_duplicates(df, pct, True),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


def get_retail_v2_corruptions():
    """Define corruptions for Online Retail V2."""
    return [
        Corruption("negative_qty_non_cancel", "Negative qty for non-cancelled", 8,
                   lambda df, pct, tb: corrupt_negative_values(df, pct, tb, 'Quantity')),
        Corruption("zero_prices", "Zero unit prices", 6,
                   lambda df, pct, tb: corrupt_impossible_values(df, pct, tb, 'UnitPrice', [0])),
        Corruption("extreme_prices", "Extreme prices ($10K+)", 3,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, tb, 'UnitPrice', 10000, 999999)),
        Corruption("future_dates", "Invoice dates in 2026", 4,
                   lambda df, pct, tb: corrupt_future_timestamps(df, pct, tb, 'InvoiceDate')),
        Corruption("invalid_customer_ids", "Invalid customer IDs", 7,
                   lambda df, pct, tb: corrupt_invalid_codes(df, pct, tb, 'CustomerID', ['GUEST', 'UNKNOWN', 'TEST', '-1', 'NULL', '0'])),
        Corruption("math_errors", "LineTotal calculation errors", 5,
                   lambda df, pct, tb: corrupt_math_errors(df, pct, tb)),
        Corruption("country_code_mix", "Mixed country codes/names", 6,
                   lambda df, pct, tb: corrupt_country_codes(df, pct, tb)),

        # Rare
        Corruption("duplicate_invoices", "Duplicate invoice numbers", 1,
                   lambda df, pct, tb: corrupt_duplicate_with_conflict(df, pct, tb, 'InvoiceNo', {}),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed - holiday sale glitch
        Corruption("holiday_glitch", "Period of zero prices", 2,
                   lambda df, pct, tb: corrupt_impossible_values(df, pct, True, 'UnitPrice', [0]),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


def get_airquality_v2_corruptions():
    """Define corruptions for Air Quality V2."""
    return [
        Corruption("date_format_chaos", "Mixed date formats", 25,
                   lambda df, pct, tb: corrupt_timestamp_format(df, pct, tb, 'Date Local')),
        Corruption("future_dates", "Dates in 2026", 3,
                   lambda df, pct, tb: corrupt_future_timestamps(df, pct, tb, 'Date Local')),
        Corruption("coord_precision_loss", "Rounded coordinates", 12,
                   lambda df, pct, tb: corrupt_coordinate_precision(df, pct, tb)),
        Corruption("state_name_mismatch", "State name doesn't match code", 8,
                   lambda df, pct, tb: corrupt_code_mismatch(df, pct, tb, 'State Name')),
        Corruption("exact_duplicates", "Complete row duplicates", 4,
                   lambda df, pct, tb: corrupt_exact_duplicates(df, pct, tb)),
        Corruption("near_duplicates", "Near-duplicate readings", 5,
                   lambda df, pct, tb: corrupt_near_duplicates(df, pct, tb, {'Arithmetic Mean': (-2, 2), 'AQI': (-5, 5)})),
        Corruption("cbsa_inconsistency", "CBSA name formatting", 10,
                   lambda df, pct, tb: corrupt_case_inconsistency(df, pct, tb, 'CBSA Name')),
        Corruption("county_mismatch", "County name doesn't match code", 6,
                   lambda df, pct, tb: corrupt_code_mismatch(df, pct, tb, 'County Name')),

        # Rare
        Corruption("extreme_reading", "Extreme AQI reading", 0.5,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, tb, 'AQI', 400, 600),
                   corruption_type=CorruptionType.RARE),

        # Timeboxed - wildfire season
        Corruption("wildfire_season", "Period of high readings", 5,
                   lambda df, pct, tb: corrupt_extreme_outliers(df, pct, True, 'Arithmetic Mean', 50, 200),
                   corruption_type=CorruptionType.TIMEBOXED),
    ]


# =============================================================================
# MAIN CORRUPTION ENGINE
# =============================================================================

def apply_corruptions(df: pd.DataFrame, corruptions: List[Corruption], args) -> pd.DataFrame:
    """Apply selected corruptions to a dataframe."""
    # Filter corruptions based on options
    available = [c for c in corruptions if c.corruption_type == CorruptionType.STANDARD]

    if args.include_rare:
        available += [c for c in corruptions if c.corruption_type == CorruptionType.RARE]

    if args.include_timeboxed:
        available += [c for c in corruptions if c.corruption_type == CorruptionType.TIMEBOXED]

    # Randomize selection if requested
    if args.randomize:
        n_corruptions = random.randint(1, min(8, len(available)))
        selected = random.sample(available, n_corruptions)
        print(f"    Randomly selected {n_corruptions} corruptions")
    else:
        selected = available

    # Apply each corruption
    for corruption in selected:
        # Scale percentage
        pct = corruption.base_percentage * args.scale

        # Determine if this specific corruption should be timeboxed
        is_timeboxed = corruption.corruption_type == CorruptionType.TIMEBOXED

        try:
            df = corruption.apply_fn(df, pct, is_timeboxed)
            n_estimated = int(len(df) * (pct / 100))
            print(f"    Applied {corruption.name}: ~{n_estimated:,} rows ({pct:.1f}%)")
        except Exception as e:
            print(f"    WARNING: Failed to apply {corruption.name}: {e}")

    return df


def corrupt_dataset(name: str, load_fn, corruptions_fn, args, target_rows: int = None, is_placebo: bool = False):
    """Generic function to corrupt a dataset."""
    print(f"\nProcessing {name}...")

    # Load data
    df = load_fn()
    if target_rows and len(df) > target_rows:
        df = df.sample(n=target_rows, random_state=args.seed)
    print(f"  Loaded {len(df):,} rows")

    # If this is a designated placebo dataset, skip all corruption
    if is_placebo:
        print(f"  PLACEBO DATASET - no corruptions applied")
        output_path = os.path.join(args.output_dir, f"{name}.csv")
        df.to_csv(output_path, index=False)
        print(f"  Saved {len(df):,} rows to {output_path}")
        return df

    # Save placebo version if requested (for non-placebo datasets)
    if args.placebo:
        placebo_path = os.path.join(args.output_dir, f"{name}_placebo.csv")
        df.to_csv(placebo_path, index=False)
        print(f"  Saved placebo to {placebo_path}")

    # Get and apply corruptions
    corruptions = corruptions_fn()
    df = apply_corruptions(df, corruptions, args)

    # Shuffle and save
    df = df.sample(frac=1, random_state=args.seed + 1).reset_index(drop=True)
    output_path = os.path.join(args.output_dir, f"{name}.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df):,} rows to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Corrupt datasets with realistic data quality issues")
    parser.add_argument('--randomize', action='store_true',
                        help='Randomly select 1-8 corruptions per dataset')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale corruption percentages (default: 1.0)')
    parser.add_argument('--placebo', action='store_true',
                        help='Also generate clean placebo versions')
    parser.add_argument('--include-rare', action='store_true',
                        help='Include rare corruptions (0.5-1%% rate)')
    parser.add_argument('--include-timeboxed', action='store_true',
                        help='Include timeboxed corruptions (contiguous blocks)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='corrupted',
                        help='Output directory (default: corrupted/)')
    parser.add_argument('--datasets', type=str, nargs='*',
                        help='Specific datasets to process (default: all)')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("DATA QUALITY POC - Corruption Engine")
    print("="*70)
    print(f"\nOptions:")
    print(f"  Randomize: {args.randomize}")
    print(f"  Scale: {args.scale}")
    print(f"  Placebo: {args.placebo}")
    print(f"  Include rare: {args.include_rare}")
    print(f"  Include timeboxed: {args.include_timeboxed}")
    print(f"  Seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")

    # Define all datasets
    datasets = {
        'usgs_earthquakes_v1': {
            'load': lambda: pd.read_csv('raw/usgs_earthquakes.csv'),
            'corruptions': lambda: [],  # NO CORRUPTIONS - pure placebo
            'target_rows': 50000,
            'is_placebo': True,
        },
        'nyc_taxi_v1': {
            'load': lambda: pd.read_csv('raw/nyc_taxi_1m.csv'),
            'corruptions': get_taxi_v1_corruptions,
            'target_rows': 900000,
        },
        'online_retail_v1': {
            'load': lambda: pd.read_csv('raw/online_retail.csv'),
            'corruptions': get_retail_v1_corruptions,
            'target_rows': None,
        },
        'chicago_crimes_v1': {
            'load': lambda: pd.read_csv('raw/chicago_crimes_500k.csv'),
            'corruptions': get_crimes_v1_corruptions,
            'target_rows': 393000,
        },
        'air_quality_v1': {
            'load': lambda: pd.read_csv('raw/air_quality_full.csv', low_memory=False),
            'corruptions': get_airquality_v1_corruptions,
            'target_rows': 250000,
        },
        'chicago_crimes_v2': {
            'load': lambda: pd.read_csv('raw/chicago_crimes.csv'),
            'corruptions': get_crimes_v2_corruptions,
            'target_rows': 100000,
        },
        'nyc_taxi_v2': {
            'load': lambda: pd.read_csv('raw/nyc_taxi_jan2024.csv'),
            'corruptions': get_taxi_v2_corruptions,
            'target_rows': 50000,
        },
        'online_retail_v2': {
            'load': lambda: pd.read_csv('raw/online_retail.csv'),
            'corruptions': get_retail_v2_corruptions,
            'target_rows': 25000,
        },
        'air_quality_v2': {
            'load': lambda: pd.read_csv('raw/air_quality_full.csv', low_memory=False),
            'corruptions': get_airquality_v2_corruptions,
            'target_rows': 10000,
        },
    }

    # Filter datasets if specified
    if args.datasets:
        datasets = {k: v for k, v in datasets.items() if k in args.datasets}

    # Check for raw files
    required_files = ['raw/nyc_taxi_1m.csv', 'raw/online_retail.csv',
                      'raw/chicago_crimes_500k.csv', 'raw/air_quality_full.csv',
                      'raw/chicago_crimes.csv', 'raw/nyc_taxi_jan2024.csv',
                      'raw/usgs_earthquakes.csv']
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\nERROR: Missing raw files: {missing}")
        print("Run 'python download_datasets.py' first.")
        return 1

    # Process each dataset
    for name, config in datasets.items():
        try:
            corrupt_dataset(
                name,
                config['load'],
                config['corruptions'],
                args,
                config.get('target_rows'),
                config.get('is_placebo', False)
            )
        except Exception as e:
            print(f"ERROR processing {name}: {e}")

    print("\n" + "="*70)
    print("CORRUPTION COMPLETE")
    print("="*70)

    if args.placebo:
        print("\nPlacebo files generated alongside corrupted versions.")
        print("Use these as a control group for comparison.")

    return 0


if __name__ == "__main__":
    exit(main())
