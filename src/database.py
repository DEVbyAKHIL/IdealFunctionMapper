"""
This Handles all the data preparation and database setup
I have put all the data loading and saving here so it will be easy to run it first.
"""

from .data_handler import TrainingDataHandler, IdealDataHandler, TestDataHandler, DataLoadError

def prepare_database():
    """
    Loads all the CSV files and saves them into the SQLite database
    """
    train_handler = TrainingDataHandler()
    ideal_handler = IdealDataHandler()
    test_handler = TestDataHandler()

    try:
        train_df = train_handler.load()
        train_handler.save_to_db(train_df, "training_data")
    except DataLoadError as e:
        print(f"Error in loading the training data: {e}")

    try:
        ideal_df = ideal_handler.load()
        ideal_handler.save_to_db(ideal_df, "ideal_functions")
    except DataLoadError as e:
        print(f"Error in loading ideal data: {e}")

    try:
        test_df = test_handler.load()
        test_handler.save_to_db(test_df, "test_data")
    except DataLoadError as e:
        print(f"Error in loading test data: {e}")

if __name__ == "__main__":
    prepare_database()
