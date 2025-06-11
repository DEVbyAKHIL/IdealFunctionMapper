import unittest
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from src.function_mapper import find_best_fit_functions
from src.database import load_all_data
from src.test_mapper import calculate_max_deviation

class TestFunctionMapping(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///data/data.db")
        cls.train_df, cls.ideal_df, cls.test_df, cls.engine = load_all_data()

    def test_find_best_fit_functions_returns_dict(self):
        result = find_best_fit_functions(self.train_df, self.ideal_df)
        print("Best fit result:", result)  # Add this line for debugging
        self.assertIsInstance(result, dict)
        self.assertTrue(all(key.startswith("y") for key in result.keys()))
        self.assertTrue(all(value.startswith("y") for value in result.values()))

    def test_calculate_max_deviation_structure(self):
        best_fits = {
            'y1': 'y42',
            'y2': 'y41',
            'y3': 'y43',
            'y4': 'y48'
        }
        max_dev = calculate_max_deviation(self.train_df, self.ideal_df, best_fits)
        self.assertIsInstance(max_dev, dict)
        self.assertEqual(set(max_dev.keys()), set(best_fits.values()))

    def test_calculate_max_deviation_values(self):
        best_fits = {
            'y1': 'y42',
            'y2': 'y41',
            'y3': 'y43',
            'y4': 'y48'
        }
        max_dev = calculate_max_deviation(self.train_df, self.ideal_df, best_fits)
        for value in max_dev.values():
            self.assertIsInstance(value, float)
            self.assertGreaterEqual(value, 0.0)

if __name__ == "__main__":
    unittest.main()
