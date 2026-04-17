# 📊 Ideal Function Mapper (Colab Implementation)

## 📌 Project Overview

This project implements a **data analysis pipeline in Python using Google Colab** to:

* Select best-fit ideal functions from a dataset of 50 functions
* Compare them with training datasets
* Map test data points using a deviation-based rule
* Store results in a database
* Visualize outputs using plots

The entire implementation is contained in a **single Jupyter Notebook (`main.ipynb`)**.

---

## 🎯 Objective

The objective of this project is to:

* Identify **4 ideal functions** that best fit the training data using **Sum of Squared Errors (SSE)**
* Assign test data points using a **√2 deviation threshold**
* Ensure reliable and consistent mapping of new data points

---

## 🗂️ Project Structure

```
IdealFunctionMapper/
│
├── data/
│   ├── train.csv
│   ├── ideal.csv
│   └── test.csv
│
├── src/
│   └── main.ipynb

```

---

## ⚙️ Technologies Used

* **Python 3**
* **Pandas**
* **NumPy**
* **SQLAlchemy**
* **SQLite**
* **Bokeh**
* **Google Colab / Jupyter Notebook**

---

## 🔄 Workflow

### 1. Data Loading

* Load CSV files (`train.csv`, `ideal.csv`, `test.csv`)
* Convert into Pandas DataFrames

### 2. Data Preprocessing

* Handle missing values
* Ensure consistent numeric types
* Align X values

### 3. Ideal Function Selection

* Compute **SSE (Sum of Squared Errors)**
* Select the best-fit ideal function for each training dataset

### 4. Deviation Calculation

* Compute deviation:

  ```
  Δy = |y_test - y_ideal|
  ```

### 5. Mapping Rule

* Assign test points only if:

  ```
  Δy ≤ max_deviation × √2
  ```

### 6. Database Storage

* Store results in **SQLite database**
* Includes:

  * Training data
  * Ideal functions
  * Mapping results

### 7. Visualization

* Training data plots
* Ideal function plots
* Comparison plots
* Test data mapping results

---

## 🧪 Unit Testing

Basic unit tests are implemented inside the notebook to verify:

* SSE calculation
* Maximum deviation
* √2 threshold rule
* Ideal function selection
* Database creation
* Mapping correctness

---

## 🚀 How to Run

### ▶️ Option 1: Google Colab (Recommended)

1. Open `main.ipynb` in Google Colab
2. Upload the dataset files (`train.csv`, `ideal.csv`, `test.csv`)
3. Run all cells sequentially

---

### 💻 Option 2: Local Jupyter Notebook

```bash
pip install pandas numpy sqlalchemy bokeh
jupyter notebook
```

Open:

```
src/main.ipynb
```

---

## 📊 Outputs

The notebook generates:

* Function comparison plots
* Scatter plots of test data
* Assigned vs unassigned data points
* SQLite database (`database.db`)

---

## 📌 Key Concepts

* Least Squares Method (SSE)
* Deviation-based mapping
* Data preprocessing
* Database integration
* Data visualization

---

## 📖 Future Improvements

* Add advanced regression models
* Improve outlier handling
* Introduce adaptive thresholds
* Enhance visualization interactivity

---

## 👤 Author

**Akhil Thoppil Shabu**
Matriculation Number: 4250982

IU International University of Applied Sciences

---

---
