"""Create data/sample_sales.csv with a few deliberate problems to clean:
missing values, duplicate rows and extra spaces. Run: python data/generate_sample.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 300

catalog = {
    "Laptop": ("Electronics", 900),
    "Phone": ("Electronics", 600),
    "Headphones": ("Electronics", 120),
    "Desk": ("Furniture", 250),
    "Chair": ("Furniture", 150),
    "Notebook": ("Stationery", 5),
    "Pen Set": ("Stationery", 12),
}
products = rng.choice(list(catalog), size=n)

df = pd.DataFrame(
    {
        "order_id": np.arange(1001, 1001 + n),
        "order_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "region": rng.choice(["Cairo", "Alexandria", "Giza", "Aswan"], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "product": products,
        "category": [catalog[p][0] for p in products],
        "quantity": rng.integers(1, 6, n),
        "unit_price": [catalog[p][1] * rng.uniform(0.9, 1.1) for p in products],
        "discount": rng.choice([0.0, 0.05, 0.1, 0.2], size=n, p=[0.5, 0.2, 0.2, 0.1]),
        "customer_age": rng.integers(18, 66, n).astype(float),
    }
)
df["unit_price"] = df["unit_price"].round(2)
df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
df = df.sort_values("order_date").reset_index(drop=True)

# Add problems on purpose
df.loc[rng.choice(n, 15, replace=False), "customer_age"] = np.nan
df.loc[rng.choice(n, 10, replace=False), "discount"] = np.nan
df.loc[rng.choice(n, 12, replace=False), "region"] = df["region"] + " "
df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)

out = Path(__file__).parent / "sample_sales.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} rows")
