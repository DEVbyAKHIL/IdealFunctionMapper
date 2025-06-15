"""
Unit tests for testing function and test mapping logic
I have added a simple code for unittesting because I'm still learning unittest, but I have tried to test the main logic
"""

import unittest
from src.function_mapper import FunctionMapper
from src.test_mapper import TestMapper

class TestFunctionMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.func_mapper = FunctionMapper()
        cls.test_mapper = TestMapper()
        cls.train, cls.ideal = cls.func_mapper.load_data()
        cls.mapping = cls.func_mapper.find_best_fit_functions(cls.train, cls.ideal)

    def test_find_best_fit_functions(self):
        mapping = self.func_mapper.find_best_fit_functions(self.train, self.ideal)
        self.assertIsInstance(mapping, dict)
        self.assertTrue(all(isinstance(k, str) for k in mapping.keys()))
        self.assertTrue(all(isinstance(v, str) for v in mapping.values()))
        # I added this lines, to check that all four training functions are mapped
        self.assertEqual(len(mapping), 4)

    def test_max_deviation(self):
        devs = self.test_mapper.max_deviation(self.train, self.ideal, self.mapping)
        self.assertIsInstance(devs, dict)
        for v in devs.values():
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.0)

    def test_error_on_missing_file(self):
        # This is a negative test: try loading a file that doesn't exist
        from src.data_handler import DataLoadError, DataHandler
        handler = DataHandler(db_path="data/data.db")
        with self.assertRaises(DataLoadError):
            handler.load_csv("data/does_not_exist.csv")

if __name__ == "__main__":
    unittest.main()
