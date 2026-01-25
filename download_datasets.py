#!/usr/bin/env python3
"""
Download raw datasets from their public sources.

Sources:
- NYC Yellow Taxi: NYC TLC Trip Record Data
- Online Retail: UCI ML Repository
- Chicago Crimes: Chicago Data Portal (Socrata)
- EPA Air Quality: EPA AQS Pre-Generated Data Files
"""

import os
import requests
import zipfile
import io
from urllib.parse import urljoin

# Create raw directory
os.makedirs('raw', exist_ok=True)


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress indication."""
    print(f"  Downloading from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                print(f"\r  Progress: {downloaded:,} / {total_size:,} bytes ({pct:.1f}%)", end='', flush=True)
    print()


def download_nyc_taxi():
    """
    Download NYC Yellow Taxi data from TLC.
    Using January 2024 parquet file, converting to CSV.
    """
    print("\n" + "="*60)
    print("NYC Yellow Taxi Data")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas required. Run: pip install pandas pyarrow")
        return False

    # January 2024 Yellow Taxi data
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    parquet_path = "raw/nyc_taxi_2024_01.parquet"

    if not os.path.exists(parquet_path):
        download_file(url, parquet_path)
    else:
        print(f"  Using cached {parquet_path}")

    # Convert to CSV and create samples
    print("  Converting to CSV...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows")

    # Full January file for v2 (smaller dataset)
    csv_path = "raw/nyc_taxi_jan2024.csv"
    if not os.path.exists(csv_path):
        sample = df.sample(n=min(100000, len(df)), random_state=42)
        sample.to_csv(csv_path, index=False)
        print(f"  Saved {len(sample):,} rows to {csv_path}")

    # 1M sample for v1 (larger dataset)
    csv_path_1m = "raw/nyc_taxi_1m.csv"
    if not os.path.exists(csv_path_1m):
        if len(df) >= 1000000:
            sample_1m = df.sample(n=1000000, random_state=42)
        else:
            # If not enough rows, download more months
            print("  Need more data for 1M sample, downloading February...")
            url2 = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet"
            parquet_path2 = "raw/nyc_taxi_2024_02.parquet"
            if not os.path.exists(parquet_path2):
                download_file(url2, parquet_path2)
            df2 = pd.read_parquet(parquet_path2)
            df_combined = pd.concat([df, df2], ignore_index=True)
            sample_1m = df_combined.sample(n=1000000, random_state=42)
        sample_1m.to_csv(csv_path_1m, index=False)
        print(f"  Saved {len(sample_1m):,} rows to {csv_path_1m}")

    return True


def download_online_retail():
    """
    Download Online Retail dataset from UCI ML Repository.
    """
    print("\n" + "="*60)
    print("Online Retail Data (UCI)")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas required. Run: pip install pandas openpyxl")
        return False

    csv_path = "raw/online_retail.csv"
    if os.path.exists(csv_path):
        print(f"  Using cached {csv_path}")
        return True

    # UCI hosts this as an Excel file
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    xlsx_path = "raw/online_retail.xlsx"

    if not os.path.exists(xlsx_path):
        download_file(url, xlsx_path)

    print("  Converting Excel to CSV...")
    df = pd.read_excel(xlsx_path)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df):,} rows to {csv_path}")

    return True


def download_chicago_crimes():
    """
    Download Chicago Crimes data from Chicago Data Portal (Socrata API).
    """
    print("\n" + "="*60)
    print("Chicago Crimes Data")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas required. Run: pip install pandas")
        return False

    # Socrata API endpoint for Chicago Crimes
    # Using $limit and $offset for pagination
    base_url = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

    # Download 100K for v2
    csv_path = "raw/chicago_crimes.csv"
    if not os.path.exists(csv_path):
        print("  Downloading 100K rows for v2...")
        url = f"{base_url}?$limit=103000&$order=date DESC"
        download_file(url, csv_path)
        df = pd.read_csv(csv_path)
        print(f"  Downloaded {len(df):,} rows to {csv_path}")
    else:
        print(f"  Using cached {csv_path}")

    # Download 500K for v1
    csv_path_500k = "raw/chicago_crimes_500k.csv"
    if not os.path.exists(csv_path_500k):
        print("  Downloading 500K rows for v1 (this may take a while)...")
        all_data = []
        offset = 0
        limit = 50000
        target = 500000

        while offset < target:
            url = f"{base_url}?$limit={limit}&$offset={offset}&$order=date DESC"
            print(f"\r  Fetching rows {offset:,} - {offset+limit:,}...", end='', flush=True)
            response = requests.get(url)
            response.raise_for_status()

            chunk_df = pd.read_csv(io.StringIO(response.text))
            if len(chunk_df) == 0:
                break
            all_data.append(chunk_df)
            offset += limit

        print()
        df = pd.concat(all_data, ignore_index=True)
        df.to_csv(csv_path_500k, index=False)
        print(f"  Saved {len(df):,} rows to {csv_path_500k}")
    else:
        print(f"  Using cached {csv_path_500k}")

    return True


def download_air_quality():
    """
    Download EPA Air Quality data (PM2.5) from EPA AQS.
    """
    print("\n" + "="*60)
    print("EPA Air Quality Data (PM2.5)")
    print("="*60)

    try:
        import pandas as pd
    except ImportError:
        print("  ERROR: pandas required. Run: pip install pandas")
        return False

    csv_path = "raw/air_quality_full.csv"
    if os.path.exists(csv_path):
        print(f"  Using cached {csv_path}")
        return True

    # EPA provides annual zip files with daily PM2.5 data
    # Using 2023 data
    url = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_2023.zip"
    zip_path = "raw/daily_pm25_2023.zip"

    if not os.path.exists(zip_path):
        download_file(url, zip_path)

    print("  Extracting ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the CSV inside
        csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
        print(f"  Extracting {csv_name}...")
        zf.extract(csv_name, 'raw/')
        os.rename(f'raw/{csv_name}', csv_path)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Extracted {len(df):,} rows to {csv_path}")

    return True


def main():
    print("="*60)
    print("DATA QUALITY POC - Downloading Raw Datasets")
    print("="*60)
    print("\nThis script downloads datasets from their original public sources.")
    print("Total download size is approximately 500MB-1GB.")
    print()

    results = {}

    results['nyc_taxi'] = download_nyc_taxi()
    results['online_retail'] = download_online_retail()
    results['chicago_crimes'] = download_chicago_crimes()
    results['air_quality'] = download_air_quality()

    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\nAll datasets downloaded successfully!")
        print("Run 'python corrupt_datasets.py' to generate corrupted versions.")
    else:
        print("\nSome downloads failed. Check errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
