"""
This program Finds the best-fit ideal functions for each training functions
This is the part where I used the least squares method to find the closest match.
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

class FunctionMapper:
    """
    Maps the training functions to ideal functions using least squares method
    """
    def __init__(self, db_path="data/data.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")

    def load_data(self):
        """
        it Loads training, ideal data from the database
        """
        train = pd.read_sql("SELECT * FROM training_data", self.engine)
        ideal = pd.read_sql("SELECT * FROM ideal_functions", self.engine)
        return train, ideal

    def find_best_fit_functions(self, train, ideal):
        """
        Returns a mapping: training column -> ideal column.
        I looped over each training function and find the ideal function with the smallest squared error.
        """
        mapping = {}
        for col in train.columns:
            if col == "x":
                continue
            min_err = None
            best = None
            for ic in ideal.columns:
                if ic == "x":
                    continue
                err = np.sum((train[col] - ideal[ic])**2)
                # print(f"Comparing {col} to {ic}, error: {err}")  # Debug
                if (min_err is None) or (err < min_err):
                    min_err = err
                    best = ic
            mapping[col] = best
        return mapping

if __name__ == "__main__":
    mapper = FunctionMapper()
    train, ideal = mapper.load_data()
    mapping = mapper.find_best_fit_functions(train, ideal)
    print("Best fit mapping is (training -> ideal):")
    print(mapping)
