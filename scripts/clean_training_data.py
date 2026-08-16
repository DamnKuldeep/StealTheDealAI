import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

RAW_PATH = settings.DATA_RAW_DIR / "amazon_india_30k.csv"  # Update to actual Kaggle file name if needed
OUTPUT_PATH = settings.DATA_PROCESSED_DIR / "training_data.csv"


def parse_price(price_series: pd.Series) -> pd.Series:
    # Strip everything but digits and '.', e.g. '₹1,299.00' -> '1299.00'
    cleaned = price_series.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')


def clean_data():
    if not RAW_PATH.exists():
        print(f"File not found: {RAW_PATH}")
        print("Please download the PromptCloud Amazon India dataset and place it here.")
        return

    print("Loading raw training data...")
    df = pd.read_csv(RAW_PATH, low_memory=False)

    print(f"Original size: {len(df)}")

    # Use Mrp (undiscounted list price) as the training target, not Price (the scraped
    # listing's current, possibly-already-discounted price) - falls back to Price where
    # Mrp is missing. We're training a "true value" estimator: if it learned to predict
    # Price, it would learn to predict already-discounted prices, undermining the whole
    # point of a steal-deal detector. Mirrors the actual_price/discount_price choice in
    # clean_rag_data.py.
    if 'Mrp' in df.columns:
        df['true_price'] = df['Mrp'].fillna(df.get('Price'))
    else:
        df['true_price'] = df['Price']

    # Drop rows without a product title or price
    df = df.dropna(subset=['Product Title', 'true_price'])

    # Parse prices
    df['parsed_price'] = parse_price(df['true_price'])

    # Filter valid prices in range MIN_PRICE - MAX_PRICE
    df = df[df['parsed_price'].notna()]
    df = df[(df['parsed_price'] >= settings.MIN_PRICE) & (df['parsed_price'] <= settings.MAX_PRICE)]

    # Deduplicate by ASIN if available, else by Title
    if 'Product Asin' in df.columns:
        df = df.drop_duplicates(subset=['Product Asin'])
    else:
        df = df.drop_duplicates(subset=['Product Title'])

    print(f"Filtered size: {len(df)}")

    # fillna BEFORE astype(str): pandas' modern string dtype preserves NaN through
    # astype(str) (unlike legacy object dtype, which stringified it to "nan") - a NaN
    # left in any of these would silently null out the whole concatenated field below.
    title = df['Product Title'].fillna('').astype(str).str.strip()
    category = (
        df.get('Category', pd.Series('Unknown', index=df.index)).fillna('Unknown').astype(str)
        .str.split('|').str[-1].str.strip()
    )
    brand = df.get('Brand', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()
    desc = df.get('Product Description', pd.Series('', index=df.index)).fillna('').astype(str).str.strip().str[:300]
    details = df.get('Pack Size Or Quantity', pd.Series('', index=df.index)).fillna('').astype(str).str.strip()

    out_df = pd.DataFrame({
        'title': title,
        'category': category,
        'brand': brand,
        'description': title + ' ' + category + ' ' + brand + ' ' + desc,
        'full_text': (
            'Title: ' + title
            + '\nCategory: ' + category
            + '\nBrand: ' + brand
            + '\nDescription: ' + desc
            + '\nDetails: ' + details
        ),
        'price': df['parsed_price'],
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved {len(out_df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_data()
