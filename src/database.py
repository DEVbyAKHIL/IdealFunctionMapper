import os
import pandas as pd
from sqlalchemy import create_engine

def load_all_data():
    """Load training, ideal, test data, and engine — for full use in main + testing."""
    engine = create_engine("sqlite:///data/data.db")
    train_df = pd.read_sql("SELECT * FROM training_data", engine)
    ideal_df = pd.read_sql("SELECT * FROM ideal_functions", engine)
    test_df = pd.read_sql("SELECT * FROM test_data", engine)
    return train_df, ideal_df, test_df, engine

def load_csv_to_df(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path} successfully.")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def store_df_to_sqlite(df, table_name, db_path='data/data.db'):
    """Stores a DataFrame into a SQLite database."""
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Stored {table_name} in {db_path}.")
    except Exception as e:
        print(f"Error storing {table_name}: {e}")

def process_all_files():
    """Loads and stores all 3 CSVs into SQLite database."""
    base_path = "data"

    train_df = load_csv_to_df(os.path.join(base_path, "train.csv"))
    if train_df is not None:
        store_df_to_sqlite(train_df, "training_data")

    ideal_df = load_csv_to_df(os.path.join(base_path, "ideal.csv"))
    if ideal_df is not None:
        store_df_to_sqlite(ideal_df, "ideal_functions")

    test_df = load_csv_to_df(os.path.join(base_path, "test.csv"))
    if test_df is not None:
        store_df_to_sqlite(test_df, "test_data")

if __name__ == "__main__":
    process_all_files()
