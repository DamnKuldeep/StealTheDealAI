import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

RAW_PATH = settings.DATA_RAW_DIR / "Amazon-Products.csv"  # Lokesh Parab dataset's pre-merged file
OUTPUT_PATH = settings.DATA_PROCESSED_DIR / "rag_catalog.csv"


def parse_price(price_series: pd.Series) -> pd.Series:
    cleaned = price_series.astype(str).str.replace(r'[^\d.]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')


def clean_rag_data():
    if not RAW_PATH.exists():
        print(f"File not found: {RAW_PATH}")
        print("Please download the Lokesh Parab Amazon Products dataset and place it here.")
        return

    print("Loading raw RAG data...")
    df = pd.read_csv(RAW_PATH, low_memory=False)

    print(f"Original size: {len(df)}")

    # Needs 'actual_price' and 'name'
    if 'actual_price' not in df.columns:
        # Some versions have 'discount_price', we can use that if actual_price is missing
        df['actual_price'] = df.get('discount_price', df.get('price'))

    df = df.dropna(subset=['name', 'actual_price'])

    df['parsed_price'] = parse_price(df['actual_price'])

    # Filter valid prices MIN_PRICE - MAX_PRICE
    df = df[df['parsed_price'].notna()]
    df = df[(df['parsed_price'] >= settings.MIN_PRICE) & (df['parsed_price'] <= settings.MAX_PRICE)]

    # Drop duplicates
    df = df.drop_duplicates(subset=['name', 'main_category'])

    print(f"Filtered size: {len(df)}")

    # fillna BEFORE astype(str): pandas' modern string dtype preserves NaN through
    # astype(str) (unlike legacy object dtype, which stringified it to "nan") - a NaN
    # left in any of these would silently null out the whole concatenated document_text.
    name = df['name'].fillna('').astype(str).str.strip()
    main_cat = df.get('main_category', pd.Series('Unknown', index=df.index)).fillna('Unknown').astype(str).str.strip()
    sub_cat = df.get('sub_category', pd.Series('Unknown', index=df.index)).fillna('Unknown').astype(str).str.strip()
    rating = df.get('ratings', pd.Series('No rating', index=df.index)).fillna('No rating').astype(str)

    out_df = pd.DataFrame({
        'name': name,
        'category': main_cat,
        'sub_category': sub_cat,
        'document_text': name + ' | Category: ' + main_cat + ' > ' + sub_cat + ' | Rating: ' + rating,
        'price': df['parsed_price'],
        'ratings': rating,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved {len(out_df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_rag_data()
