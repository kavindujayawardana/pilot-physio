"""
Sanity check for ingestion outputs.

This script verifies that cleaned session-level parquet files:
- can be loaded successfully,
- contain expected columns,
- include valid Event label distributions.

This script is not part of the main analysis pipeline and is used
only for data validation during development.
"""


import pandas as pd

df = pd.read_parquet("processed/5/5_CA_clean.parquet")
print(df.head())
print(df.columns)
print(df["Event"].value_counts())
