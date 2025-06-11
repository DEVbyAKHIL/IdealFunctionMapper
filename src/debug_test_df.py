import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///data/data.db")
test_df = pd.read_sql("SELECT * FROM test_data", engine)

print("[INFO] Columns in test_df:", test_df.columns.tolist())
print(test_df.head())
