"""
Data handler classes for loading and saving the data
I used inheritance for avoiding the repeatation of code for each data types (training, ideal, test)
"""

import pandas as pd
from sqlalchemy import create_engine
from .exceptions import DataLoadError

class DataHandler:
    """
    Base class for loading and saving data to SQLite
    I wanted to make the DB path flexible, so it's in an argument
    """
    def __init__(self, db_path="data/data.db"):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def load_csv(self, filepath):
        """
        it Loads a CSV files into a pandas DataFrame
        shows DataLoadError if the file can't be loaded
        """
        try:
            df = pd.read_csv(filepath)
            # print(f"Loaded {filepath}, shape: {df.shape}")  # Debugging line
            return df
        except Exception as e:
            # If we want to know exactly what went wrong, It shows my custom error.
            raise DataLoadError(f"Failed to load {filepath}: {e}")

    def save_to_db(self, df, table_name):
        """
        This Saves a DataFrame to the SQLite database
        """
        try:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            # print(f"Saved the table {table_name} to DB.")  # Debugging line
        except Exception as e:
            print(f"Couldn't save table {table_name}: {e}")

class TrainingDataHandler(DataHandler):
    """
    This Handles the loading and saving of training data
    """
    def load(self):
        # Used this file for training data
        return self.load_csv("data/train.csv")

class IdealDataHandler(DataHandler):
    """
    It Handles the loading and saving of ideal function data
    """
    def load(self):
        return self.load_csv("data/ideal.csv")

class TestDataHandler(DataHandler):
    """
    Handles loading and saving of the test data.
    """
    def load(self):
        return self.load_csv("data/test.csv")

    def preview(self):
        """
        I used this to check the file quickly.
        """
        df = self.load()
        print(df.head())
