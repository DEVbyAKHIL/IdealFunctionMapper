import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data():
    """Load training and ideal function data from SQLite database."""
    engine = create_engine("sqlite:///data/data.db")
    train_df = pd.read_sql("SELECT * FROM training_data", engine)
    ideal_df = pd.read_sql("SELECT * FROM ideal_functions", engine)
    return train_df, ideal_df

import numpy as np

def find_best_fit_functions(train_df, ideal_df):
    """
    Find the best-fitting ideal function for each training function based on least squares.
    Returns a dictionary mapping training function column to best matching ideal function column.
    """
    best_fits = {}

    # Ensure we're comparing on same X values
    if 'x' not in train_df.columns or 'x' not in ideal_df.columns:
        raise ValueError("Both train_df and ideal_df must contain an 'x' column.")

    # Keep only function columns (excluding 'x')
    train_columns = [col for col in train_df.columns if col.lower() != 'x']
    ideal_columns = [col for col in ideal_df.columns if col.lower() != 'x']

    # Compare each training function to all ideal functions
    for train_col in train_columns:
        min_error = float('inf')
        best_fit_col = None

        for ideal_col in ideal_columns:
            try:
                # Compute least squares error
                error = np.sum((train_df[train_col] - ideal_df[ideal_col]) ** 2)
                if error < min_error:
                    min_error = error
                    best_fit_col = ideal_col
            except Exception as e:
                print(f"Error comparing {train_col} to {ideal_col}: {e}")

        # Double check it's valid
        if best_fit_col and best_fit_col.lower().startswith('y'):
            best_fits[train_col] = best_fit_col
        else:
            print(f"⚠️ Warning: No valid best fit for {train_col}")
            best_fits[train_col] = "INVALID"

    return best_fits


def main():
    """Main function to load data and print best fit mappings."""
    train_df, ideal_df = load_data()
    best_fits = find_best_fit_functions(train_df, ideal_df)

    print("Best fit mappings (Training → Ideal):")
    for train_func, ideal_func in best_fits.items():
        print(f"{train_func} -> {ideal_func}")

if __name__ == "__main__":
    main()
