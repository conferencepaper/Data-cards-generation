import pandas as pd
import heapq

import pandas as pd
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

nl = NormalizedLevenshtein()
def improved_custom_df_summary(df, unique_limit=5):
    """
    Terse summary per column.
    
    Numeric:
      - "Stats": min,25%,med,mean,std,75%,max.
      - "Rep": e.g., "float, 2dp".
    
    Datetime:
      - "Range": start->end.
      - "Rep": "YYYY-MM-DD".
    
    Object:
      - "Uniq": unique count.
      - "Top": top 5 values as percentages.
      - "Rep": Diverse sample and most dissimilar.
      
    Header format: Col '<name>' [dtype]: Tot: N | Miss: M (X%)
    """
    lines = []
    for col in df.columns:
        total = len(df[col])
        missing = df[col].isna().sum()
        miss_pct = (missing / total * 100) if total else 0
        header = f"Col '{col}' [{df[col].dtype}]: Miss: {missing} ({miss_pct:.0f}%)"
        lines.append(header)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            stats = {
                "min": round(desc.get("min", float('nan')), 2),
                "25%": round(desc.get("25%", float('nan')), 2),
                "med": round(desc.get("50%", float('nan')), 2),
                "mean": round(desc.get("mean", float('nan')), 2),
                "std": round(desc.get("std", float('nan')), 2),
                "75%": round(desc.get("75%", float('nan')), 2),
                "max": round(desc.get("max", float('nan')), 2)
            }
            stats_str = ", ".join(f"{k}:{v}" for k,v in stats.items())
            #can be improven it 
            #rep = "Rep: float, 2dp" if any(isinstance(x, float) for x in df[col].dropna()) else "Rep: int"
            lines.append(f"  Stats: {stats_str}")
            #lines.append(f"  {rep}")
        
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            rng = f"{df[col].min()} -> {df[col].max()}"
            lines.append(f"  Range: {rng}")
            #lines.append("  Rep: YYYY-MM-DD")
        

        else:
            unique_vals = list(df[col].dropna().unique())
            num_unique = len(unique_vals)
            # Top 5 values as percentages.
            vc = df[col].value_counts(normalize=True) * 100
            top = vc.head(unique_limit)
            top_str = ", ".join(f"'{val}':{pct:.1f}%" for val, pct in top.items())
            lines.append(f" Uniq: {num_unique}  Top: {top_str}")
            
            # Representation: from the remaining values (unused), select a diverse subset.
            used_vals = list(top.keys())
            unused_vals = [val for val in unique_vals if val not in used_vals]

            if len(unused_vals)>5:
                # For each unused value, compute the total normalized distance to each used value.
                levi_sums = [sum(nl.distance(val, used) for used in used_vals) for val in unused_vals]
                # Create a list of tuples: (distance_sum, index, value)
                indexed_levi = [(levi_sums[index], index, unused_vals[index]) for index in range(len(unused_vals))]
                # Select up to 5 values with the smallest total distance (most similar to the used group)
                low_5 = heapq.nsmallest(5, indexed_levi)
                low_5_values = [tup[2] for tup in low_5]
                low_val = ", ".join(f"{x}" for x in low_5_values)
                lines.append(f"  Rep: Unused diverse: {low_val}")

            elif  1 <len(unused_vals)<5:
                low_val=", ".join(f"{x}" for x in unused_vals)
                lines.append(f"  Rep: Unused diverse: {low_val}")

    
    return "\n".join(lines) 
def create_prompt_description_target(
    dataframe: pd.DataFrame,
    dataset_name: str = "",
    include_stats: bool = False,
    use_real_column_names: bool = True,
    rows_to_display: int = 5,
    target_column: str = None
) -> str:
    """
    Generates a prompt that instructs an LLM to create a structured data card
    in JSON format, containing exactly seven specific keys.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame for which to generate the prompt.
    dataset_name : str, optional
        Name of the dataset. Defaults to an empty string.
    include_stats : bool, optional
        Whether to include basic descriptive statistics in the prompt.
        Defaults to False.
    use_real_column_names : bool, optional
        If False, replaces column names with generic ones like 'column_1', etc.
        Defaults to True.
    rows_to_display : int, optional
        Number of top rows to display as a preview in the prompt. Defaults to 5.
    target_column : str, optional
        Name of the target/label column in the dataset, if applicable. Defaults to None.

    Returns
    -------
    str
        A formatted prompt string for an LLM, specifying the desired JSON structure.
    """
    # Make a safe copy of the DataFrame for any modifications
    preview_df = dataframe.copy()

    # Optionally replace column names with generic names
    if not use_real_column_names:
        renamed_cols = {
            old_name: f"column_{i+1}" for i, old_name in enumerate(preview_df.columns)
        }
        preview_df.rename(columns=renamed_cols, inplace=True)
        
        # Update target_column if it exists in the new names
        if target_column and target_column in renamed_cols:
            target_column = renamed_cols[target_column]

    # Calculate total rows and the percentage of rows displayed
    total_rows = len(dataframe)
    percent_display = round((rows_to_display * 100) / total_rows, 2) if total_rows > 0 else 0

    # Generate CSV string for the first few rows
    data_head_str = preview_df.head(rows_to_display).to_csv(index=False)

    # Optionally include descriptive statistics
    data_stats_str = ""
    if include_stats:
        # Assumes you have a custom function `improved_custom_df_summary` defined elsewhere
        stats_df = improved_custom_df_summary(dataframe)
        data_stats_str = "\n*Statistical Summary:*\n" + stats_df

    # Build the initial prompt
    prompt = f"""You are a data scientist helping to analyze the {dataset_name} dataset.
This dataset has a total of {total_rows} rows.

Below is a small preview of the data (first {percent_display}% rows):
{data_head_str}
{data_stats_str}"""

    # Add target column details if applicable
    if target_column:
        prompt += (
            f"\n\n**Target Column**: The target column for this dataset is '{target_column}'. "
            "This column is typically used as the label for predictive modeling or as the main variable of interest."
        )

    # Add the instructions for the LLM
    # Note the clarified structure for ColumnsDescription
    prompt += """**Your Task:**

1. **Columns Description**
   For each column in the dataset, return an object with the following keys:
   - **Type**: The underlying data type of the column (possible values: "integer", "float", "string").
   - **Domain**: An array representing the allowed values or range. For categorical columns, list the acceptable strings; for numeric columns, provide [min, max].
   - **FormCanonical**: A precise canonical representation with formatting rules. For example, "Integer value without decimals representing age in years" or "Lowercase, trimmed string with fixed length".
   - **Purpose**: A brief explanation of what the column represents.
   - **Identifier**: A designation for the column’s role. Possible values: "Measure", "Enumerated", "Binary", "Date", "Attribute", "Target".

2. **Data Instance**
   Provide a concise explanation of what a single record (row) in this dataset represents.

3. **Dataset Description**
   Write a short, high-level overview of what the dataset represents and its main purpose.

4. **Title**
   Propose a concise and clear title for this dataset.

5. **Dataset Usage**
   Describe potential use cases or scenarios for which this dataset could be applied.

6. **Rule Definition**
   Identify data quality rules with the following categories (use this exact structure):

   - **FunctionalDependencies**
     - *Description*: Array of objects
       - **determinant**: Attributes that uniquely determine others
       - **dependent**: Attributes determined by the determinant

   - **ConditionalFunctionalDependencies**
     - *Description*: Array of objects
       - **determinant**: List of attributes
       - **dependent**: List of attributes
       - **condition**: The condition under which this dependency holds

   - **DenialConstraints**
     - *Description*: Array of objects
       - **constraint**: A rule or condition that must never occur

   - **OtherConstraints**
     - *Description*: Array or object for any additional constraints not covered above

7. **Fairness**
   Identify any protected or sensitive variables in the dataset (e.g., race, gender, age). For each variable, specify:
   - Potentially privileged groups
   - Potentially unprivileged groups
   - Possible biases or fairness considerations

**Output Requirements:**

- Return your answer in JSON format only.
- The JSON must have exactly seven top-level keys:
  1. "ColumnsDescription": an object where each key is a column name and each value is an object with the keys "Type", "Domain", "FormCanonical", "Purpose", and "Identifier".
  2. "DataInstance": a string.
  3. "DatasetDescription": a string.
  4. "DatasetUsage": a string.
  5. "Title": a string.
  6. "Rule_def": an object with exactly four keys: "FunctionalDependencies", "ConditionalFunctionalDependencies", "DenialConstraints", "OtherConstraints".
  7. "Fairness": an object with two keys: "favorable_labels" (an array of favorable values) and "protected_attributes" (an array of objects, each with "feature" and "reference_group", where "reference_group" is an array specifying the privileged group).

Do not provide any additional fields or commentary.
"""

    return prompt.strip()
