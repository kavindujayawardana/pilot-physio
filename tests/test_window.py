import numpy as np
import pandas as pd

df = pd.read_parquet("windows/22/windows.parquet")
pointer = df["raw_data_pointer"].iloc[0]

x = np.load(pointer)
print(x.shape, x.dtype)

