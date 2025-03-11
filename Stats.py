import pandas as pd
import numpy as np

def numeric_stats(series):
    """
    Generates an enhanced set of descriptive statistics for a numeric column.
    """
    # Convert column to numeric, coercing invalid values to NaN
    col = pd.to_numeric(series, errors='coerce')

    # Drop NaNs for numeric computation
    valid_col = col.dropna()

    # Basic descriptive stats
    count_valid = valid_col.shape[0]
    unique_count = valid_col.nunique()
    mean_val = round(valid_col.mean(), 2)
    std_val = round(valid_col.std(), 2)
    min_val = valid_col.min()
    q25 = valid_col.quantile(0.25)
    median_val = round(valid_col.median(), 2)
    q75 = valid_col.quantile(0.75)
    max_val = valid_col.max()

    # Extended stats
    missing_count = series.shape[0] - count_valid
    value_range = max_val - min_val
    iqr = q75 - q25
    skew_val = round(valid_col.skew(), 4)  # More decimal precision if desired
    kurt_val = round(valid_col.kurt(), 4)

    # Most frequent (top 5 modes)
    mode_values = valid_col.mode().tolist()
    mode_values = mode_values[:5]

    stats = {
        "count": count_valid,
        "missing_count": missing_count,
        "unique_count": unique_count,
        "Mean": mean_val,
        "Std. Deviation": std_val,
        "Min": min_val,
        "25%": q25,
        "50% (Median)": median_val,
        "75%": q75,
        "Max": max_val,
        "Range": value_range,
        "IQR": iqr,
        "Skewness": skew_val,
        "Kurtosis": kurt_val,
        "Most Frequent": mode_values
    }

    return stats

def alphanumeric_stats(series):
    """
    Generates an enhanced set of descriptive statistics for an alphanumeric/object column.
    """
    column_data = series.dropna().astype(str)

    # Basic statistics
    unique_count = column_data.nunique()
    most_frequent_counts = column_data.value_counts().head(5)
    most_frequent = most_frequent_counts.to_dict()

    # String length statistics
    string_lengths = column_data.str.len()
    mean_length = string_lengths.mean()
    max_length = string_lengths.max()
    min_length = string_lengths.min()

    # Character composition
    contains_letters = column_data.str.contains(r'[a-zA-Z]').mean() * 100
    contains_digits = column_data.str.contains(r'\d').mean() * 100

    # Missing and blank values
    missing_count = series.isnull().sum()
    blank_count = (column_data == "").sum()

    # Case analysis
    uppercase_count = column_data.str.isupper().sum()
    lowercase_count = column_data.str.islower().sum()

    # Prefix and suffix analysis
    common_prefix_counts = column_data.str[:2].value_counts().head(5)
    common_prefix = common_prefix_counts.to_dict()

    # For suffix, you might want to standardize the length. 
    # Here we use last 3 characters as in your example.
    common_suffix_counts = column_data.str[-3:].value_counts().head(5)
    common_suffix = common_suffix_counts.to_dict()

    # Alpha/numeric ratios
    letter_count = column_data.str.count(r'[a-zA-Z]')
    digit_count = column_data.str.count(r'\d')
    # Avoid division by zero by adding 1 to digit_count
    alpha_numeric_ratio = (letter_count / (digit_count + 1)).mean()

    stats = {
        "unique_count": unique_count,
        "most_frequent": most_frequent,
        "mean_length": mean_length,
        "max_length": max_length,
        "min_length": min_length,
        "contains_letters_percent": contains_letters,
        "contains_digits_percent": contains_digits,
        "missing_count": missing_count,
        "blank_count": blank_count,
        "uppercase_count": uppercase_count,
        "lowercase_count": lowercase_count,
        "common_prefix": common_prefix,
        "common_suffix": common_suffix,
        "alpha_numeric_ratio": alpha_numeric_ratio
    }

    return stats

def custom_describe(df):
    """
    Returns descriptive statistics for each column in a DataFrame using either
    numeric_stats or alphanumeric_stats, depending on the column's dtype.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    dict
        A dictionary where each key is the column name and each value is a 
        dictionary of descriptive stats from either numeric_stats or alphanumeric_stats.
    """
    results = {}
    for col in df.columns:
        col_dtype = df[col].dtype
        # Check if it's numeric
        if pd.api.types.is_numeric_dtype(col_dtype):
            results[col] = numeric_stats(df[col])
        else:
            results[col] = alphanumeric_stats(df[col])
    
    return results
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Function to create a table with histograms under column headers
def create_table_with_histograms(df):
    fig = go.Figure()

    # Add columns to the table with histograms
    headers = []
    cells = []
    for column in df.columns:
        headers.append(column)

        # Add histogram as a subfigure
        if pd.api.types.is_numeric_dtype(df[column]):
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=df[column], nbinsx=10, marker_color="blue"))
            hist_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=60)
            fig.add_trace(
                go.Histogram(x=df[column], nbinsx=10, name=column, marker_color="blue")
            )

        cells.append(df[column].astype(str).tolist())

    # Add table structure
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color="lightgrey", align="center"),
            cells=dict(values=cells, fill_color="white", align="center"),
        )
    )

    fig.update_layout(width=1000, height=600)
    return fig


import pandas as pd
import matplotlib.pyplot as plt
"""
def plot_histogram_alphanumeric(df, column, bins=10):
    # 1. Check if the column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # 2. Determine if this column can be considered numeric
    #    (We can also directly check df[column].dtype or try a to_numeric() conversion)
    if pd.api.types.is_numeric_dtype(df[column]):
        # This is a numeric column
        data = df[column].dropna()
        if data.empty:
            print(f"No numeric data to plot for column '{column}'.")
            return
        # Plot histogram
        plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {column} (Numeric)")
        plt.show()
    else:

        value_counts = df[column].dropna().astype(str).value_counts().head(10)
        plt.bar(value_counts.index, value_counts.values)
        plt.xticks(rotation=45)
        plt.title(f"Bar Plot of Top 10 Values in '{column}'")
        plt.show()
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram_alphanumeric(df, column, bins=10):
    """
    Plots a histogram for a given column in a DataFrame. 
      - If the column is numeric, it plots the numeric values.
      - If the column is alphanumeric (object/string), 
        it plots the distribution of string lengths.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    column : str
        The name of the column to plot.
    bins : int, optional
        Number of histogram bins for numeric data or string length, 
        by default 10.
    """

    # Create a new figure and axes
    fig, ax = plt.subplots()

    if column not in df.columns:
        ax.text(0.5, 0.5, f"Column '{column}' not found", 
                ha='center', va='center', fontsize=12)
        return fig
    # 1. Check if the column exists
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    # 2. Determine if this column is numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        # This is a numeric column
        data = df[column].dropna()
        if data.empty:
            print(f"No numeric data to plot for column '{column}'.")
            return
        # Plot histogram
        ax.hist(data, bins=bins, color='skyblue', edgecolor='black')
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {column} (Numeric)")
    else:
        # For alphanumeric columns, we plot the top 10 values
        value_counts = df[column].dropna().astype(str).value_counts().head(10)
        ax.bar(value_counts.index, value_counts.values)
        # Rotate x-axis labels properly:
        ax.tick_params(axis='x', rotation=45)
        ax.set_title(f"Bar Plot of Top 10 Values in '{column}'")

    fig.tight_layout()
    return fig


