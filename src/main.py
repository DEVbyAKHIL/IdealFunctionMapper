from src.database import prepare_database
from src.function_mapper import FunctionMapper
from src.test_mapper import TestMapper
from src.visualize import visualize

def main():
    prepare_database()
    mapper = FunctionMapper()
    train, ideal = mapper.load_data()
    mapping = mapper.find_best_fit_functions(train, ideal)
    test_mapper = TestMapper()
    train, ideal, test = test_mapper.load_data()
    devs = test_mapper.max_deviation(train, ideal, mapping)
    result_df = test_mapper.map_test_points(test, ideal, mapping, devs)
    test_mapper.save_results(result_df)
    visualize()

if __name__ == "__main__":
    main()
