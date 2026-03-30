
!pip -q install sqlalchemy bokeh pandas

# ============================================================
# 1) IMPORTS
# ============================================================
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
import numpy as np

# ============================================================
# 2) CUSTOM EXCEPTIONS
# ============================================================
class DataValidationError(Exception):
    """Raised when input data does not match required format/constraints."""


class MappingError(Exception):
    """Raised when mapping cannot be performed due to missing ideal x-values or other issues."""


# ============================================================
# 3) OOP STRUCTURE (Inheritance)
# ============================================================
@dataclass
class DatasetBase:
    """Base dataset class (for inheritance)."""
    path: str

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        if df.empty:
            raise DataValidationError(f"{self.path} loaded but is empty.")
        return df


@dataclass
class TrainingDataset(DatasetBase):
    """Training dataset: expects x + y1..y4."""
    def validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # Normalize column names to lowercase
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        required = ["x", "y1", "y2", "y3", "y4"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"Training CSV missing columns: {missing}. Found: {list(df.columns)}")

        # Ensure numeric
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[required].isna().any().any():
            raise DataValidationError("Training CSV has non-numeric or missing values.")

        # Sort by x
        df = df.sort_values("x").reset_index(drop=True)
        return df


@dataclass
class IdealDataset(DatasetBase):
    """Ideal dataset: expects x + y1..y50."""
    def validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        if "x" not in df.columns:
            raise DataValidationError("Ideal CSV must have column 'x'.")

        y_cols = [f"y{i}" for i in range(1, 51)]
        missing = [c for c in y_cols if c not in df.columns]
        if missing:
            raise DataValidationError(f"Ideal CSV missing columns: {missing[:5]}... total missing={len(missing)}")

        # numeric
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        for c in y_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if df[["x"] + y_cols].isna().any().any():
            raise DataValidationError("Ideal CSV has non-numeric or missing values.")

        df = df.sort_values("x").reset_index(drop=True)
        return df


@dataclass
class TestDataset(DatasetBase):
    """Test dataset: expects x + y."""
    def validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]

        required = ["x", "y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"Test CSV missing columns: {missing}. Found: {list(df.columns)}")

        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        if df[required].isna().any().any():
            raise DataValidationError("Test CSV has non-numeric or missing values.")

        df = df.sort_values("x").reset_index(drop=True)
        return df


# ============================================================
# 4) DATABASE MANAGER (SQLAlchemy)
# ============================================================
class DatabaseManager:
    """Handles SQLite database creation and table writes using SQLAlchemy."""
    def __init__(self, db_path: str = "assignment.db"):
        self.db_path = db_path
        self.engine: Engine = create_engine(f"sqlite:///{db_path}", echo=False)

    def write_table(self, df: pd.DataFrame, table_name: str) -> None:
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)

    def read_table(self, table_name: str) -> pd.DataFrame:
        return pd.read_sql_table(table_name, self.engine)


# ============================================================
# 5) CORE LOGIC: Select best ideal functions + map test points
# ============================================================
def compute_sse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sum of squared errors."""
    diff = y_true - y_pred
    return float(np.sum(diff * diff))


def compute_max_abs_dev(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Max absolute deviation."""
    return float(np.max(np.abs(y_true - y_pred)))


class IdealFunctionSelector:
    """Selects best ideal function for each training function by least squares."""
    def __init__(self, train_df: pd.DataFrame, ideal_df: pd.DataFrame):
        self.train_df = train_df
        self.ideal_df = ideal_df

        # Ensure x alignment (merge on x)
        merged = pd.merge(train_df, ideal_df, on="x", how="inner", suffixes=("", "_ideal"))
        if merged.empty:
            raise DataValidationError("No overlapping x-values between training and ideal datasets.")
        self.merged = merged

    def select_best_for_each_training(self) -> Dict[str, Dict[str, float]]:
        """
        Returns dict like:
        {
          "y1": {"ideal_col": "yXX", "ideal_index": XX, "sse": ..., "max_dev": ...},
          ...
        }
        """
        results = {}
        ideal_cols = [f"y{i}" for i in range(1, 51)]

        for train_col in ["y1", "y2", "y3", "y4"]:
            y_train = self.merged[train_col].to_numpy()

            best = None  # (sse, ideal_col, max_dev)
            for ic in ideal_cols:
                y_ideal = self.merged[ic].to_numpy()
                sse = compute_sse(y_train, y_ideal)
                if (best is None) or (sse < best[0]):
                    max_dev = compute_max_abs_dev(y_train, y_ideal)
                    best = (sse, ic, max_dev)

            sse, ideal_col, max_dev = best
            results[train_col] = {
                "ideal_col": ideal_col,
                "ideal_index": int(ideal_col.replace("y", "")),
                "sse": float(sse),
                "max_dev": float(max_dev),
            }

        return results


class TestMapper:
    """Maps test points to one of the chosen ideal functions under sqrt(2) rule."""
    def __init__(self, ideal_df: pd.DataFrame, chosen: Dict[str, Dict[str, float]]):
        self.ideal_df = ideal_df
        self.chosen = chosen

        # Build lookup: x -> row (fast)
        self.ideal_by_x = ideal_df.set_index("x")

    def map_test_points(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Output columns:
          x, y, delta_y, ideal_func_no
        ideal_func_no can be NaN if unassigned.
        """
        sqrt2 = math.sqrt(2)
        rows = []

        chosen_ideals = []
        for train_col, info in self.chosen.items():
            chosen_ideals.append((train_col, info["ideal_col"], info["ideal_index"], info["max_dev"]))

        for _, r in test_df.iterrows():
            x_val = float(r["x"])
            y_test = float(r["y"])

            if x_val not in self.ideal_by_x.index:
                # Can't compute ideal y at this x => unassigned
                rows.append({"x": x_val, "y": y_test, "delta_y": np.nan, "ideal_func_no": np.nan})
                continue

            best_match = None  # (delta, ideal_index)
            for train_col, ideal_col, ideal_idx, max_dev in chosen_ideals:
                y_ideal = float(self.ideal_by_x.loc[x_val, ideal_col])
                delta = abs(y_test - y_ideal)
                threshold = float(max_dev) * sqrt2

                if delta <= threshold:
                    if (best_match is None) or (delta < best_match[0]):
                        best_match = (delta, ideal_idx)

            if best_match is None:
                rows.append({"x": x_val, "y": y_test, "delta_y": np.nan, "ideal_func_no": np.nan})
            else:
                rows.append({"x": x_val, "y": y_test, "delta_y": best_match[0], "ideal_func_no": best_match[1]})

        return pd.DataFrame(rows)


# ============================================================
# 6) VISUALIZATION (Bokeh)
# ============================================================



class Visualizer:
    """Creates 6 aligned Bokeh plots for the assignment."""

    def __init__(self, train_df, ideal_df, chosen, mapped_df):
        self.train_df = train_df
        self.ideal_df = ideal_df
        self.chosen = chosen
        self.mapped_df = mapped_df

    def build(self, output_html="visualization.html"):

        output_file(output_html, title="IU Assignment - 6 Graphs")

        # ----------------------------
        # GRAPH 1: Training functions
        # ----------------------------
        p1 = figure(width=1000, height=300,
                    title="(1) Training Functions",
                    x_axis_label="x", y_axis_label="y")

        for col in ["y1","y2","y3","y4"]:
            p1.line(self.train_df["x"], self.train_df[col],
                    line_width=2, legend_label=col)

        p1.legend.click_policy = "hide"

        # ----------------------------
        # GRAPH 2: Chosen Ideal Functions
        # ----------------------------
        p2 = figure(width=1000, height=300,
                    title="(2) Chosen Ideal Functions",
                    x_axis_label="x", y_axis_label="y")

        for train_col, info in self.chosen.items():
            ideal_col = info["ideal_col"]
            p2.line(self.ideal_df["x"],
                    self.ideal_df[ideal_col],
                    line_width=2,
                    line_dash="dashed",
                    legend_label=f"{ideal_col} (for {train_col})")

        p2.legend.click_policy = "hide"

        # ----------------------------
        # GRAPH 3: Training + Ideal Overlay
        # ----------------------------
        p3 = figure(width=1000, height=300,
                    title="(3) Training vs Ideal Overlay",
                    x_axis_label="x", y_axis_label="y")

        for col in ["y1","y2","y3","y4"]:
            p3.line(self.train_df["x"], self.train_df[col],
                    line_width=2)

        for train_col, info in self.chosen.items():
            ideal_col = info["ideal_col"]
            p3.line(self.ideal_df["x"],
                    self.ideal_df[ideal_col],
                    line_width=2,
                    line_dash="dashed")

        # ----------------------------
        # GRAPH 4: Test Points (Raw)
        # ----------------------------
        p4 = figure(width=1000, height=300,
                    title="(4) Test Points (Raw)",
                    x_axis_label="x", y_axis_label="y")

        p4.circle(self.mapped_df["x"],
                  self.mapped_df["y"],
                  size=6)



# ----------------------------

p = figure(width=1000, height=400,
           title="Test Points: Assigned vs Unassigned",
           x_axis_label="x", y_axis_label="y")

mapped_tmp = mapped_df.copy()
assigned_mask = mapped_tmp["ideal_func_no"].notna()

# Unassigned points
p.scatter(mapped_tmp.loc[~assigned_mask, "x"],
          mapped_tmp.loc[~assigned_mask, "y"],
          size=6, marker="circle",
          legend_label="unassigned")

# Assigned points
p.scatter(mapped_tmp.loc[assigned_mask, "x"],
          mapped_tmp.loc[assigned_mask, "y"],
          size=6, marker="circle",
          legend_label="assigned")

p.legend.click_policy = "hide"

save(p)
show(p)

print(" Saved final_graphs.html")



# ============================================================
# 7) MAIN PIPELINE (All steps)
# ============================================================
@dataclass
class Config:
    train_csv: str = "/content/train.csv"
    ideal_csv: str = "/content/ideal.csv"
    test_csv: str = "/content/test.csv"
    db_path: str = "assignment.db"
    out_html: str = "visualization.html"


def run_pipeline(cfg: Config) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    # Load + validate
    train_ds = TrainingDataset(cfg.train_csv)
    ideal_ds = IdealDataset(cfg.ideal_csv)
    test_ds  = TestDataset(cfg.test_csv)

    train_df = train_ds.validate_and_normalize(train_ds.load())
    ideal_df = ideal_ds.validate_and_normalize(ideal_ds.load())
    test_df  = test_ds.validate_and_normalize(test_ds.load())

    # Create DB and write base tables
    db = DatabaseManager(cfg.db_path)
    db.write_table(train_df, "training_data")
    db.write_table(ideal_df, "ideal_functions")

    # Select best ideal functions for y1..y4
    selector = IdealFunctionSelector(train_df, ideal_df)
    chosen = selector.select_best_for_each_training()

    # Map test points
    mapper = TestMapper(ideal_df, chosen)
    mapped_df = mapper.map_test_points(test_df)

    # Save mapping results to DB
    db.write_table(mapped_df, "mapped_test_data")

    # Visualize
    viz = Visualizer(train_df, ideal_df, chosen, mapped_df)
    html_path = viz.build(cfg.out_html)

    print(" Done")
    print(f"DB created: {cfg.db_path}")
    print("Chosen ideal functions:")
    for k, v in chosen.items():
        print(f"  {k} -> ideal y{v['ideal_index']} | SSE={v['sse']:.4f} | max_dev={v['max_dev']:.6f}")
    print(f"Visualization saved: {html_path}")

    return chosen, mapped_df, train_df, ideal_df, test_df



cfg = Config(
    train_csv="/content/train.csv",
    ideal_csv="/content/ideal.csv",
    test_csv="/content/test.csv",
    db_path="assignment.db",
    out_html="visualization.html"
)

chosen, mapped_df, train_df, ideal_df, test_df = run_pipeline(cfg)


mapped_df.head(10)




viz = Visualizer(train_df, ideal_df, chosen, mapped_df)
viz.build("all_6_graphs.html")
print(" Saved all_6_graphs.html")

# ------------------------------------------------
UNIT TESTS
#-------------------------------------------------

chosen, mapped_df, train_df, ideal_df, test_df = run_pipeline(cfg)

import unittest
import math
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

class TestIUAssignment(unittest.TestCase):

    # -------------------------
    # 1) Math Function Tests
    # -------------------------
    def test_sse(self):
        y1 = np.array([1.0, 2.0, 3.0])
        y2 = np.array([1.0, 2.0, 4.0])
        self.assertAlmostEqual(compute_sse(y1, y2), 1.0)

    def test_max_abs_dev(self):
        y1 = np.array([1.0, 2.0, 3.0])
        y2 = np.array([0.0, 2.5, 2.0])
        self.assertAlmostEqual(compute_max_abs_dev(y1, y2), 1.0)

    def test_threshold_formula(self):
        max_dev = 2.0
        threshold = max_dev * math.sqrt(2)
        self.assertTrue(2.82 <= threshold <= 2.83)

    # -------------------------
    # 2) Chosen Ideal Checks
    # -------------------------
    def test_chosen_has_four(self):
        self.assertEqual(set(chosen.keys()), {"y1","y2","y3","y4"})

    def test_ideal_index_range(self):
        for k,v in chosen.items():
            self.assertTrue(1 <= int(v["ideal_index"]) <= 50)

    # -------------------------
    # 3) Database Tests
    # -------------------------
    def test_db_exists(self):
        self.assertTrue(os.path.exists("assignment.db"))

    def test_required_tables_exist(self):
        engine = create_engine("sqlite:///assignment.db")
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        self.assertIn("training_data", tables)
        self.assertIn("ideal_functions", tables)
        self.assertIn("mapped_test_data", tables)

    def test_training_columns(self):
        engine = create_engine("sqlite:///assignment.db")
        df = pd.read_sql_table("training_data", engine)
        self.assertEqual(list(df.columns), ["x","y1","y2","y3","y4"])

    def test_ideal_columns(self):
        engine = create_engine("sqlite:///assignment.db")
        df = pd.read_sql_table("ideal_functions", engine)
        expected = ["x"] + [f"y{i}" for i in range(1,51)]
        self.assertEqual(list(df.columns), expected)

    def test_mapped_columns(self):
        engine = create_engine("sqlite:///assignment.db")
        df = pd.read_sql_table("mapped_test_data", engine)
        self.assertEqual(list(df.columns), ["x","y","delta_y","ideal_func_no"])

    # -------------------------
    # 4) Mapping Rule Check
    # -------------------------
    def test_mapping_rule_respected(self):
        sqrt2 = math.sqrt(2)

        assigned = mapped_df.dropna(subset=["ideal_func_no","delta_y"])

        ideal_to_maxdev = {
            int(v["ideal_index"]): float(v["max_dev"])
            for v in chosen.values()
        }

        for _, row in assigned.iterrows():
            ideal_no = int(row["ideal_func_no"])
            delta = float(row["delta_y"])
            threshold = ideal_to_maxdev[ideal_no] * sqrt2

            self.assertLessEqual(delta, threshold + 1e-9)

# Run tests
unittest.main(argv=[''], exit=False)

# -------------------------------------------------------------------------

from google.colab import files
files.download("assignment.db")
files.download("visualization.html")



