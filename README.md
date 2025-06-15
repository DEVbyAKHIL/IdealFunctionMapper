
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

## ✨ Features

- 📥 **Loads training, ideal, and test data from CSV files**
- 🗄️ **Stores all data in a SQLite database with SQLAlchemy**
- 🧮 **Selects four ideal functions using least squares minimization**
- 🧩 **Maps test data to ideal functions if within allowed deviation**
- 📊 **Visualizes results using Bokeh (interactive HTML plot!)**
- 🏗️ **Object-oriented design with inheritance**
- 🚨 **Standard and custom exception handling**
- 🧪 **Unit tests for all main logic**
- 📝 **Fully documented with docstrings and comments**


## 📁 Project Structure

python-project/<br/>
├── data/ <br/>
│ ├── train.csv # Input training data<br/>
│ ├── ideal.csv # Ideal function data<br/>
│ ├── test.csv # Test data<br/>
│ └── data.db # SQLite database (auto-generated)<br/>
├── src/<br/>
│ ├── init.py<br/>
│ ├── data_handler.py<br/>
│ ├── exceptions.py<br/>
│ ├── function_mapper.py # Finds best-fit ideal functions<br/>
│ ├── test_mapper.py # Calculates max deviation<br/>
│ ├── visualize.py # Plots with Bokeh<br/>
│ ├── database.py # Loads CSVs into<br/>
├── tests/<br/>
  └── test_functionmapper.py<br/>

## 🚀 How to Run

1. Activate virtual environment:
    ".\venv\Scripts\activate"

2. Install requirements:
    "pip install -r requirements.txt"

3. Load CSVs into SQLite:
     "python -m src.database"
   
5. Run main logic:
   "python -m src.function_mapper"

6. Map test data to ideal functions
    "python -m src.test_mapper"

7. Visualize
   "python -m src.visualize"

8.  Run all tests
    "python -m unittest discover -s tests"




---

## 🧑‍💻 Git Workflow 


- git clone -b develop -repo-url-<br/>
- cd -repo-folder-<br/>
- git checkout -b my-feature<br/>

- made changes<br/>

- git add .<br/>
- git commit -m "Added new feature"<br/>
- git push origin my-feature<br/>



---

## 💡 Notes & Tips

- All code is my own work, inspired by the IU course book and examples.
- The project is designed for clarity and learning, with personal comments and explanations.
- Input data files (`train.csv`, `ideal.csv`, `test.csv`) are **not included**
- The full source code is included in the appendix of my written assignment.

---

## 📚 References

- [📘 IU Course Book]
- [🐼 pandas docs]
- [🦎 SQLAlchemy docs]
- [📈 Bokeh docs]

---

## 🙏 Thank you for checking out my project!

