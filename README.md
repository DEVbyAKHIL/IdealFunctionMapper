# IdealFunctionMapper
A Python project that selects ideal functions based on least squares fitting, maps test data using deviation thresholds, and stores results in a SQLite database. Includes data visualization with Bokeh, OOP design, exception handling, unit tests, and full documentation.

# Python Function Mapping Project – DLMDSPWP01

## 🎯 Objective
This project maps training functions to their best-fit ideal functions using the **least squares method**. It evaluates new test data against these mappings, identifies the best matches, and stores everything in a **SQLite database**. Visual results are rendered using **Bokeh**.

## 🔧 Tech Stack

- Python 3.11
- pandas
- SQLAlchemy
- Bokeh
- SQLite
- unittest (for testing)

## 📁 Project Structure

python-project/<br/>
├── data/ <br/>
│ ├── train.csv # Input training data<br/>
│ ├── ideal.csv # Ideal function data<br/>
│ ├── test.csv # Test data<br/>
│ └── data.db # SQLite database (auto-generated)<br/>
├── src/<br/>
│ ├── function_mapper.py # Finds best-fit ideal functions<br/>
│ ├── test_mapper.py # Calculates max deviation<br/>
│ ├── visualize.py # Plots with Bokeh<br/>
│ ├── database.py # Loads CSVs into<br/>

## 🚀 How to Run

1. Activate virtual environment:
    ".\venv\Scripts\activate"

2. Install requirements:
    "pip install -r requirements.txt"

3. Load CSVs into SQLite:
     "python src/database.py"
   
5. Run main logic:
   "python src/function_mapper.py"

6. Visualize
   "python src/visualize.py"

7.  Run all tests
    "python -m unittest discover -s tests"


