"""
The ideal functions and their matched test points are Visualized using Bokeh.
I used scatter instead of circle because of the deprecation warning.
"""

import pandas as pd
from sqlalchemy import create_engine
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10

def visualize():
    """
    Plots the ideal functions and test point matches.
    """
    engine = create_engine("sqlite:///data/data.db")
    ideal = pd.read_sql("SELECT * FROM ideal_functions", engine)
    test_map = pd.read_sql("SELECT * FROM test_mapping", engine)
    output_file("visualization.html")
    p = figure(title="Mapped Test Points and Ideal Functions", x_axis_label="X", y_axis_label="Y", width=1000, height=600)
    palette = Category10[10]
    funcs = test_map["Ideal_Function"].unique()
    color_map = {}
    for i, f in enumerate(funcs):
        if f not in ideal.columns:
            continue
        color = palette[i % len(palette)]
        color_map[f] = color
        p.line(ideal["x"], ideal[f], legend_label=f, line_width=2, color=color)
    for f in funcs:
        pts = test_map[test_map["Ideal_Function"] == f]
        source = ColumnDataSource(data={"x": pts["X"], "y": pts["Y"]})
        # Using scatter instead of circle because of deprecation warning.
        p.scatter("x", "y", size=8, source=source, color=color_map.get(f, "black"), legend_label=f"Test Matches ({f})")
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    show(p)

if __name__ == "__main__":
    visualize()
