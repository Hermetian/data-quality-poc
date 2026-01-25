#!/usr/bin/env python3
"""
Generate clean datasets and corrupt them in various realistic ways.
This simulates real-world data quality issues encountered in production.
"""

import csv
import random
import os
from datetime import datetime, timedelta
from decimal import Decimal
import string

random.seed(42)  # Reproducibility

# Create directories
os.makedirs('clean', exist_ok=True)
os.makedirs('corrupted', exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def write_csv(filepath, headers, rows):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def write_csv_raw(filepath, lines):
    """Write raw lines for encoding/format corruption"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def write_csv_mixed_encoding(filepath, lines, corrupt_indices):
    """Write with mixed encodings"""
    with open(filepath, 'wb') as f:
        for i, line in enumerate(lines):
            if i in corrupt_indices:
                # Try latin-1, fall back to utf-8 if chars not encodable
                try:
                    f.write(line.encode('latin-1'))
                except UnicodeEncodeError:
                    f.write(line.encode('utf-8'))
            else:
                f.write(line.encode('utf-8'))
            f.write(b'\n')

# =============================================================================
# DATASET 1: E-COMMERCE ORDERS
# =============================================================================

def generate_ecommerce_orders():
    print("Generating e-commerce orders dataset...")

    customers = [f"CUST_{i:05d}" for i in range(1, 501)]
    products = [
        ("PROD_001", "Wireless Mouse", 29.99),
        ("PROD_002", "Mechanical Keyboard", 89.99),
        ("PROD_003", "USB-C Hub", 45.00),
        ("PROD_004", "Monitor Stand", 34.50),
        ("PROD_005", "Webcam HD", 79.99),
        ("PROD_006", "Desk Lamp", 24.99),
        ("PROD_007", "Mouse Pad XL", 19.99),
        ("PROD_008", "Cable Management Kit", 15.00),
        ("PROD_009", "Laptop Stand", 49.99),
        ("PROD_010", "Blue Light Glasses", 29.99),
    ]
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]

    headers = ["order_id", "customer_id", "product_id", "product_name", "quantity",
               "unit_price", "total_amount", "order_date", "status", "shipping_country"]

    rows = []
    base_date = datetime(2024, 1, 1)

    for i in range(1, 2001):
        product = random.choice(products)
        qty = random.randint(1, 5)
        unit_price = product[2]
        total = round(qty * unit_price, 2)
        order_date = base_date + timedelta(days=random.randint(0, 365))
        country = random.choice(["USA", "Canada", "UK", "Germany", "France", "Australia"])

        rows.append([
            f"ORD_{i:06d}",
            random.choice(customers),
            product[0],
            product[1],
            qty,
            unit_price,
            total,
            order_date.strftime("%Y-%m-%d"),
            random.choice(statuses),
            country
        ])

    write_csv('clean/ecommerce_orders.csv', headers, rows)
    return headers, rows

def corrupt_ecommerce_v1(headers, rows):
    """Corruption: Mixed date formats, type inconsistencies, missing values"""
    print("  Creating corruption v1: mixed dates, type issues, nulls...")

    corrupted = []
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y"]

    for i, row in enumerate(rows):
        new_row = list(row)

        # Random date format changes (30% of rows)
        if random.random() < 0.3:
            try:
                orig_date = datetime.strptime(row[7], "%Y-%m-%d")
                new_row[7] = orig_date.strftime(random.choice(date_formats))
            except:
                pass

        # Type inconsistencies - quantity as string sometimes
        if random.random() < 0.15:
            new_row[4] = str(row[4]) + " units"

        # Missing values (random)
        if random.random() < 0.08:
            null_col = random.choice([1, 2, 8, 9])  # customer, product, status, country
            new_row[null_col] = random.choice(["", "NULL", "N/A", None, "null"])

        # Price as string with currency symbol
        if random.random() < 0.12:
            new_row[5] = f"${row[5]}"
            new_row[6] = f"${row[6]}"

        corrupted.append(new_row)

    write_csv('corrupted/ecommerce_orders_v1_mixed_formats.csv', headers, corrupted)

def corrupt_ecommerce_v2(headers, rows):
    """Corruption: Duplicates (exact and near), referential integrity issues"""
    print("  Creating corruption v2: duplicates, integrity issues...")

    corrupted = list(rows)

    # Add exact duplicates (50 random rows)
    for _ in range(50):
        corrupted.append(list(random.choice(rows)))

    # Add near duplicates (same order_id, slightly different data)
    for _ in range(30):
        orig = list(random.choice(rows))
        # Same order_id but different quantity or status
        orig[4] = orig[4] + random.randint(1, 3)
        orig[6] = round(orig[4] * float(str(orig[5]).replace('$', '')), 2)
        corrupted.append(orig)

    # Referential integrity - invalid customer/product IDs
    for i in range(len(corrupted)):
        if random.random() < 0.05:
            corrupted[i] = list(corrupted[i])
            corrupted[i][1] = f"CUST_{random.randint(9000, 9999):05d}"  # Non-existent customer
        if random.random() < 0.05:
            corrupted[i] = list(corrupted[i])
            corrupted[i][2] = f"PROD_{random.randint(900, 999)}"  # Non-existent product

    random.shuffle(corrupted)
    write_csv('corrupted/ecommerce_orders_v2_duplicates_integrity.csv', headers, corrupted)

# =============================================================================
# DATASET 2: FINANCIAL TRANSACTIONS
# =============================================================================

def generate_financial_transactions():
    print("Generating financial transactions dataset...")

    accounts = [f"ACC{i:08d}" for i in range(1, 201)]
    tx_types = ["debit", "credit", "transfer", "fee", "refund"]
    categories = ["groceries", "utilities", "entertainment", "healthcare", "transport", "dining", "shopping", "salary"]

    headers = ["transaction_id", "account_id", "timestamp", "amount", "currency",
               "transaction_type", "category", "merchant", "balance_after"]

    rows = []
    base_time = datetime(2024, 6, 1, 8, 0, 0)

    for i in range(1, 5001):
        tx_time = base_time + timedelta(
            days=random.randint(0, 180),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        amount = round(random.uniform(1, 5000), 2)
        balance = round(random.uniform(100, 50000), 2)

        rows.append([
            f"TX{i:010d}",
            random.choice(accounts),
            tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            amount,
            "USD",
            random.choice(tx_types),
            random.choice(categories),
            f"Merchant_{random.randint(1, 500)}",
            balance
        ])

    write_csv('clean/financial_transactions.csv', headers, rows)
    return headers, rows

def corrupt_financial_v1(headers, rows):
    """Corruption: Outliers, impossible values, precision issues"""
    print("  Creating corruption v1: outliers, impossible values...")

    corrupted = []

    for i, row in enumerate(rows):
        new_row = list(row)

        # Extreme outliers (negative amounts for non-refund, huge values)
        if random.random() < 0.03:
            new_row[3] = round(random.uniform(-50000, -1), 2)  # Negative amount
        if random.random() < 0.02:
            new_row[3] = round(random.uniform(1000000, 99999999), 2)  # Absurdly high

        # Negative balance
        if random.random() < 0.04:
            new_row[8] = round(random.uniform(-10000, -1), 2)

        # Precision issues (too many decimals)
        if random.random() < 0.1:
            new_row[3] = round(float(row[3]) + 0.001234567, 9)

        # Future timestamps
        if random.random() < 0.02:
            future = datetime.now() + timedelta(days=random.randint(30, 365))
            new_row[2] = future.strftime("%Y-%m-%d %H:%M:%S")

        # Zero amounts
        if random.random() < 0.03:
            new_row[3] = 0.00

        corrupted.append(new_row)

    write_csv('corrupted/financial_transactions_v1_outliers.csv', headers, corrupted)

def corrupt_financial_v2(headers, rows):
    """Corruption: Mixed currencies, timezone chaos, scientific notation"""
    print("  Creating corruption v2: currency mix, timezone issues...")

    corrupted = []
    currencies = ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF"]

    for i, row in enumerate(rows):
        new_row = list(row)

        # Mixed currencies without conversion
        if random.random() < 0.2:
            new_row[4] = random.choice(currencies)

        # Timezone suffix chaos
        if random.random() < 0.15:
            tz_suffix = random.choice(["Z", "+00:00", "-05:00", "+08:00", " UTC", " EST", " PST"])
            new_row[2] = row[2] + tz_suffix

        # Scientific notation for large amounts
        if random.random() < 0.08:
            new_row[3] = f"{float(row[3]):.2e}"

        # Comma vs dot decimal separator
        if random.random() < 0.1:
            new_row[3] = str(row[3]).replace('.', ',')

        # Thousand separators mixed in
        if random.random() < 0.05 and float(row[3]) > 1000:
            new_row[3] = f"{float(row[3]):,.2f}"

        corrupted.append(new_row)

    write_csv('corrupted/financial_transactions_v2_formats.csv', headers, corrupted)

# =============================================================================
# DATASET 3: IOT SENSOR DATA
# =============================================================================

def generate_iot_sensors():
    print("Generating IoT sensor dataset...")

    sensors = [f"SENSOR_{i:03d}" for i in range(1, 51)]
    locations = ["Building_A", "Building_B", "Warehouse_1", "Warehouse_2", "Office_Floor_1", "Office_Floor_2"]

    headers = ["reading_id", "sensor_id", "location", "timestamp", "temperature_celsius",
               "humidity_percent", "pressure_hpa", "battery_level", "signal_strength"]

    rows = []
    base_time = datetime(2024, 9, 1, 0, 0, 0)

    for i in range(1, 10001):
        read_time = base_time + timedelta(minutes=i*5)  # Reading every 5 minutes

        # Simulate realistic sensor values with some natural variation
        temp = round(random.gauss(22, 3), 2)  # Mean 22°C, std 3
        humidity = round(random.gauss(45, 10), 2)  # Mean 45%, std 10
        pressure = round(random.gauss(1013, 5), 2)  # Mean 1013 hPa
        battery = random.randint(20, 100)
        signal = random.randint(-90, -30)  # dBm

        rows.append([
            f"READ_{i:08d}",
            random.choice(sensors),
            random.choice(locations),
            read_time.strftime("%Y-%m-%dT%H:%M:%S"),
            temp,
            humidity,
            pressure,
            battery,
            signal
        ])

    write_csv('clean/iot_sensors.csv', headers, rows)
    return headers, rows

def corrupt_iot_v1(headers, rows):
    """Corruption: Unit mixing, sensor drift, truncated records"""
    print("  Creating corruption v1: unit mixing, truncation...")

    corrupted = []

    for i, row in enumerate(rows):
        new_row = list(row)

        # Mix Celsius and Fahrenheit without indication
        if random.random() < 0.25:
            # Convert to Fahrenheit but keep same column
            new_row[4] = round(float(row[4]) * 9/5 + 32, 2)

        # Mix percent and decimal for humidity
        if random.random() < 0.15:
            new_row[5] = round(float(row[5]) / 100, 4)  # 0.45 instead of 45

        # Truncated records (missing fields)
        if random.random() < 0.05:
            cut_point = random.randint(4, 7)
            new_row = new_row[:cut_point] + [''] * (len(headers) - cut_point)

        # Sensor ID format inconsistency
        if random.random() < 0.1:
            new_row[1] = row[1].replace('SENSOR_', 'S').replace('_', '-')

        corrupted.append(new_row)

    # Add some completely malformed rows
    for _ in range(20):
        corrupted.append(["MALFORMED", "DATA", "HERE"])

    random.shuffle(corrupted)
    write_csv('corrupted/iot_sensors_v1_units_truncation.csv', headers, corrupted)

def corrupt_iot_v2(headers, rows):
    """Corruption: Out of order timestamps, gaps, duplicates with different values"""
    print("  Creating corruption v2: time series issues...")

    corrupted = []

    for i, row in enumerate(rows):
        new_row = list(row)

        # Time travel - timestamps out of order (swap with nearby)
        if random.random() < 0.1 and i > 0:
            new_row[3] = rows[max(0, i - random.randint(1, 10))][3]

        # Duplicate timestamps, different readings (sensor collision)
        if random.random() < 0.05 and i > 0:
            new_row[3] = rows[i-1][3]
            new_row[4] = round(float(row[4]) + random.uniform(-5, 5), 2)

        corrupted.append(new_row)

    # Create gaps by removing chunks
    gap_start = random.randint(1000, 2000)
    gap_size = random.randint(50, 150)
    corrupted = corrupted[:gap_start] + corrupted[gap_start + gap_size:]

    # Shuffle a portion to create ordering issues
    shuffle_start = 5000
    shuffle_end = 5500
    chunk = corrupted[shuffle_start:shuffle_end]
    random.shuffle(chunk)
    corrupted[shuffle_start:shuffle_end] = chunk

    write_csv('corrupted/iot_sensors_v2_timeseries.csv', headers, corrupted)

# =============================================================================
# DATASET 4: HEALTHCARE PATIENT VISITS
# =============================================================================

def generate_healthcare_visits():
    print("Generating healthcare visits dataset...")

    providers = [f"DR_{i:04d}" for i in range(1, 51)]
    departments = ["Cardiology", "Orthopedics", "Neurology", "Dermatology", "General Practice", "Pediatrics"]
    diagnoses = ["J06.9", "M54.5", "I10", "E11.9", "F32.9", "J45.909", "K21.0", "G43.909"]  # ICD-10 codes

    headers = ["visit_id", "patient_id", "provider_id", "department", "visit_date",
               "diagnosis_code", "diagnosis_description", "visit_duration_min",
               "copay_amount", "insurance_id", "follow_up_required"]

    diagnosis_map = {
        "J06.9": "Acute upper respiratory infection",
        "M54.5": "Low back pain",
        "I10": "Essential hypertension",
        "E11.9": "Type 2 diabetes mellitus",
        "F32.9": "Major depressive disorder",
        "J45.909": "Unspecified asthma",
        "K21.0": "Gastro-esophageal reflux disease",
        "G43.909": "Migraine, unspecified"
    }

    rows = []
    base_date = datetime(2024, 1, 1)

    for i in range(1, 3001):
        visit_date = base_date + timedelta(days=random.randint(0, 300))
        diag = random.choice(diagnoses)
        duration = random.randint(10, 60)
        copay = random.choice([0, 20, 25, 30, 40, 50, 75, 100])

        rows.append([
            f"VISIT_{i:07d}",
            f"PAT_{random.randint(1, 1000):06d}",
            random.choice(providers),
            random.choice(departments),
            visit_date.strftime("%Y-%m-%d"),
            diag,
            diagnosis_map[diag],
            duration,
            copay,
            f"INS_{random.randint(1, 50):04d}",
            random.choice(["Yes", "No"])
        ])

    write_csv('clean/healthcare_visits.csv', headers, rows)
    return headers, rows

def corrupt_healthcare_v1(headers, rows):
    """Corruption: Encoding issues, special characters, whitespace"""
    print("  Creating corruption v1: encoding, special chars, whitespace...")

    lines = [','.join(headers)]
    special_chars = ['é', 'ñ', 'ü', 'ø', 'ß', '中', '日', 'Müller', 'Señor', 'naïve']

    for i, row in enumerate(rows):
        new_row = list(row)

        # Add special characters to descriptions
        if random.random() < 0.1:
            new_row[6] = random.choice(special_chars) + " - " + str(row[6])

        # Leading/trailing whitespace
        if random.random() < 0.15:
            col = random.randint(0, len(row)-1)
            spaces = ' ' * random.randint(1, 5)
            new_row[col] = spaces + str(new_row[col]) + spaces

        # Tab characters mixed in
        if random.random() < 0.05:
            new_row[6] = str(row[6]).replace(' ', '\t')

        # Newline in field (quoted)
        if random.random() < 0.03:
            new_row[6] = f'"{row[6]}\nAdditional notes"'

        lines.append(','.join(str(x) for x in new_row))

    # Write with mixed encodings for some lines
    corrupt_indices = set(random.sample(range(len(lines)), len(lines) // 10))
    write_csv_mixed_encoding('corrupted/healthcare_visits_v1_encoding.csv', lines, corrupt_indices)

def corrupt_healthcare_v2(headers, rows):
    """Corruption: Business logic violations, impossible combinations"""
    print("  Creating corruption v2: business logic violations...")

    corrupted = []

    for i, row in enumerate(rows):
        new_row = list(row)

        # Pediatrics with adult-only conditions
        if random.random() < 0.05:
            new_row[3] = "Pediatrics"
            new_row[5] = "I10"  # Hypertension - rare in children
            new_row[6] = "Essential hypertension"

        # Zero or negative duration
        if random.random() < 0.04:
            new_row[7] = random.choice([0, -15, -30])

        # Future visit dates
        if random.random() < 0.03:
            future = datetime.now() + timedelta(days=random.randint(30, 365))
            new_row[4] = future.strftime("%Y-%m-%d")

        # Mismatched diagnosis code and description
        if random.random() < 0.08:
            # Keep code, change description to wrong one
            wrong_desc = random.choice([
                "Acute upper respiratory infection",
                "Low back pain",
                "Essential hypertension",
                "Migraine, unspecified"
            ])
            if wrong_desc != row[6]:
                new_row[6] = wrong_desc

        # Invalid diagnosis codes
        if random.random() < 0.05:
            new_row[5] = random.choice(["INVALID", "XXX.XX", "000.00", ""])

        # Copay higher than reasonable
        if random.random() < 0.02:
            new_row[8] = random.randint(500, 5000)

        corrupted.append(new_row)

    write_csv('corrupted/healthcare_visits_v2_logic.csv', headers, corrupted)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA QUALITY POC - Dataset Generator")
    print("=" * 60)
    print()

    # Generate and corrupt each dataset
    headers, rows = generate_ecommerce_orders()
    corrupt_ecommerce_v1(headers, rows)
    corrupt_ecommerce_v2(headers, rows)
    print()

    headers, rows = generate_financial_transactions()
    corrupt_financial_v1(headers, rows)
    corrupt_financial_v2(headers, rows)
    print()

    headers, rows = generate_iot_sensors()
    corrupt_iot_v1(headers, rows)
    corrupt_iot_v2(headers, rows)
    print()

    headers, rows = generate_healthcare_visits()
    corrupt_healthcare_v1(headers, rows)
    corrupt_healthcare_v2(headers, rows)
    print()

    print("=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print()
    print("Clean datasets in: ./clean/")
    print("Corrupted datasets in: ./corrupted/")
    print()
    print("Corrupted versions:")
    print("  - ecommerce_orders_v1_mixed_formats.csv: date formats, type issues, nulls")
    print("  - ecommerce_orders_v2_duplicates_integrity.csv: dupes, referential integrity")
    print("  - financial_transactions_v1_outliers.csv: outliers, impossible values")
    print("  - financial_transactions_v2_formats.csv: currency mix, timezone chaos")
    print("  - iot_sensors_v1_units_truncation.csv: unit mixing, truncated records")
    print("  - iot_sensors_v2_timeseries.csv: time series issues, gaps, ordering")
    print("  - healthcare_visits_v1_encoding.csv: encoding issues, special chars")
    print("  - healthcare_visits_v2_logic.csv: business logic violations")
