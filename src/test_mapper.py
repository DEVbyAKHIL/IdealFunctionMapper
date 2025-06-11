import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data():
    """Load all necessary datasets from SQLite database."""
    engine = create_engine("sqlite:///data/data.db")
    train_df = pd.read_sql("SELECT * FROM training_data", engine)
    ideal_df = pd.read_sql("SELECT * FROM ideal_functions", engine)
    test_df = pd.read_sql("SELECT * FROM test_data", engine)

    # Normalize column names to lowercase for consistency
    test_df.columns = test_df.columns.str.lower()

    return train_df, ideal_df, test_df, engine

def calculate_max_deviation(train_df, ideal_df, mapping):
    """Calculate the max deviation for each best-fit ideal function."""
    max_deviation = {}
    for train_col, ideal_col in mapping.items():
        deviation = np.abs(train_df[train_col] - ideal_df[ideal_col])
        max_deviation[ideal_col] = deviation.max()
    return max_deviation

def map_test_data(test_df, ideal_df, best_fit_mapping, max_deviation, engine):
    result_rows = []
    selected_ideal_cols = list(best_fit_mapping.values())

    for index, row in test_df.iterrows():
        x, y_test = row['x'], row['y']
        matched = False  # Track if this row matched any ideal function

        for ideal_col in selected_ideal_cols:
            y_ideal_row = ideal_df[ideal_df['x'] == x]

            if y_ideal_row.empty:
                continue  # x not found in ideal function

            y_ideal = y_ideal_row[ideal_col].values[0]
            delta_y = abs(y_test - y_ideal)
            threshold = max_deviation[ideal_col] * np.sqrt(2)

            print(f"[DEBUG] Test X: {x:.2f}, Y_test: {y_test:.2f}, Y_ideal ({ideal_col}): {y_ideal:.2f}, ΔY: {delta_y:.4f}, Threshold: {threshold:.4f}")

            if delta_y <= threshold:
                result_rows.append({
                    'X': x,
                    'Y': y_test,
                    'Delta_Y': delta_y,
                    'Ideal_Function': ideal_col
                })
                matched = True
                break  # Stop after first match

        if not matched:
            print(f"[INFO] No match found for test point (X={x}, Y={y_test})")

    result_df = pd.DataFrame(result_rows)

    if not result_df.empty:
        result_df.to_sql("test_mapping", engine, if_exists="replace", index=False)
        print(f"[SUCCESS] Stored {len(result_df)} matched test points in the database.")
    else:
        print("[WARNING] No test data points matched any ideal function.")

def main():
    train_df, ideal_df, test_df, engine = load_data()

    # Your mappings from function_mapper.py (manually copied)
    best_fit_mapping = {
        'y1': 'y42',
        'y2': 'y41',
        'y3': 'x',
        'y4': 'y48'
    }

    max_deviation = calculate_max_deviation(train_df, ideal_df, best_fit_mapping)
    map_test_data(test_df, ideal_df, best_fit_mapping, max_deviation, engine)

if __name__ == "__main__":
    main()
