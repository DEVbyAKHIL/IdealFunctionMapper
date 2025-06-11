import pandas as pd
from sqlalchemy import create_engine
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10

def load_data():
    engine = create_engine("sqlite:///data/data.db")
    ideal_df = pd.read_sql("SELECT * FROM ideal_functions", engine)
    test_mapping_df = pd.read_sql("SELECT * FROM test_mapping", engine)
    return ideal_df, test_mapping_df

def plot_results(ideal_df, test_mapping_df):
    output_file("visualization.html")

    p = figure(title="Mapped Test Points and Ideal Functions",
               x_axis_label='X', y_axis_label='Y', width=1000, height=600,
               tools="pan,wheel_zoom,box_zoom,reset,save,hover", tooltips="@desc")

    color_map = {}
    palette = Category10[10]  # max 10 colors
    unique_functions = test_mapping_df["Ideal_Function"].unique()

    for i, func in enumerate(unique_functions):
        if func not in ideal_df.columns:
            continue

        color = palette[i % len(palette)]
        color_map[func] = color

        p.line(ideal_df["x"], ideal_df[func], legend_label=func, line_width=2, color=color)

    # Plot test points
    for func in unique_functions:
        matched_points = test_mapping_df[test_mapping_df["Ideal_Function"] == func]
        source = ColumnDataSource(data={
            'x': matched_points['X'],
            'y': matched_points['Y'],
            'desc': [f"X={x:.2f}, Y={y:.2f}, ΔY={dy:.4f}, Match={func}" 
                     for x, y, dy in zip(matched_points['X'], matched_points['Y'], matched_points['Delta_Y'])]
        })
        p.circle('x', 'y', source=source, size=8, color=color_map[func], legend_label=f"Test Matches ({func})")

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    show(p)

def main():
    ideal_df, test_mapping_df = load_data()
    plot_results(ideal_df, test_mapping_df)

if __name__ == "__main__":
    main()
