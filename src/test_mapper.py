"""
Assigns the test data points to the closest ideal function if it's within allowed deviation
This is where I check if a test point "fits" one of the chosen ideal functions
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

class TestMapper:
    """
    Allocates test points to ideal functions using deviation thresholds.
    """
    def __init__(self, db_path="data/data.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")

    def load_data(self):
        """
        Loads training, ideal & test data from the database
        """
        train = pd.read_sql("SELECT * FROM training_data", self.engine)
        ideal = pd.read_sql("SELECT * FROM ideal_functions", self.engine)
        test = pd.read_sql("SELECT * FROM test_data", self.engine)
        test.columns = test.columns.str.lower()
        return train, ideal, test

    def max_deviation(self, train, ideal, mapping):
        """
        It Calculates the max deviation for each mapped pair.
        """
        dev = {}
        for t, i in mapping.items():
            dev[i] = np.abs(train[t] - ideal[i]).max()
        return dev

    def map_test_points(self, test, ideal, mapping, devs):
        """
        Maps each test point to an ideal function if it fits the allowed deviation.
        """
        rows = []
        for _, row in test.iterrows():
            x, y = row["x"], row["y"]
            found = False
            for ic in mapping.values():
                ideal_row = ideal[ideal["x"] == x]
                if ideal_row.empty:
                    continue
                yid = ideal_row[ic].values[0]
                d = abs(y - yid)
                if d <= devs[ic] * np.sqrt(2):
                    rows.append({"X": x, "Y": y, "Delta_Y": d, "Ideal_Function": ic})
                    found = True
                    break
            if not found:
                # I added this, if any test points are not matching (for debugging)
                # print(f"Test point ({x}, {y}) did not matched any ideal function.")
                pass
        return pd.DataFrame(rows)

    def save_results(self, df):
        """
        Saves the mapped test points to the database.
        """
        if not df.empty:
            df.to_sql("test_mapping", self.engine, if_exists="replace", index=False)
            print("Saved test_mapping table to DB.")
        else:
            print("No matches found for test data.")

if __name__ == "__main__":
    test_mapper = TestMapper()
    train, ideal, test = test_mapper.load_data()
    # For a demo, I use the real mapping. In practice, this should come from FunctionMapper.
    from src.function_mapper import FunctionMapper
    mapping = FunctionMapper().find_best_fit_functions(train, ideal)
    devs = test_mapper.max_deviation(train, ideal, mapping)
    result_df = test_mapper.map_test_points(test, ideal, mapping, devs)
    test_mapper.save_results(result_df)
