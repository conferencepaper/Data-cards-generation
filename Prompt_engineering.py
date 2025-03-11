import pandas as pd
import numpy as np
from mistralai import Mistral
import anthropic
from openai import OpenAI
import os
#Logging in with ghazzaiskander@gmail.com
#https://platform.openai.com/api-keys
open_ai_key=""
os.environ["OPENAI_API_KEY"] = open_ai_key
#https://console.mistral.ai/api-keys/
mistral_ai_key=""
os.environ["MISTRAL_API_KEY"] = mistral_ai_key
claude_key=""
#https://console.anthropic.com/settings/keys
os.environ["ANTHROPIC_API_KEY"] = claude_key  # Replace with your actual Anthropic API key

def  generate_fairness_prompt(
    dataframe: pd.DataFrame,
    dataset_name: str,
    include_stats: bool = False,
    header_variation: str = "header_csv_snippet",
    rows_to_display: int = 5,
    target_columns: str=None
) -> str:
    # Number of rows in the entire dataset
    total_rows = len(dataframe)
 
    # Generate the specified header preview using get_header_variations
    header_variations = get_header_variations(
        dataframe,
        use_real_column_names=True,
        rows_to_display=rows_to_display
    )

    # Safely get the chosen header format or default if it doesn't exist
    data_head_str = header_variations.get(header_variation, header_variations["header_json_snippet"])


    # Optionally include descriptive statistics
    data_stats_str = ""
    if include_stats:
            stats_df = dataframe.describe(include="all").fillna("")  # fill NaNs for display
            data_stats_str = "\n*Statistical Summary:*\n" + stats_df.to_string()
    prompt = f"""
You are a data scientist helping to analyze the "{dataset_name}" dataset.

This dataset has a total of {total_rows} rows.

Below is a small preview of the data (first {rows_to_display} rows):
{data_head_str}
{data_stats_str}

The target column(s) is/are: {target_columns}.

**Your Task**:
1. **Fairness**: Understand the context of bias and fairness, and how it applies to this dataset.
2. **Protected / Sensitive Variables**: Identify columns that may contain protected attributes (e.g., race, gender, age).
3. **Privileged / Unprivileged Groups**: Determine which groups in the dataset might be privileged versus unprivileged, especially within any protected attributes.
4. **Evaluation and Mitigation**: Discuss how these fairness considerations could affect model performance and decision-making. Suggest strategies or metrics (e.g., demographic parity, equal opportunity) to detect and mitigate bias.
    
**Output Requirements**:
- Provide your answers in a structured, concise format (bullet points or paragraphs).
- Focus on fairness considerations and potential bias within the dataset.
"""
    return prompt.strip()







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




import pandas as pd

def get_header_variations(df, use_real_column_names=True, rows_to_display=5):
    """
    Returns a dictionary of various ways to present the head of the DataFrame.
    Keys are referenced names, values are string representations.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame from which to show header variations.
    use_real_column_names : bool, default True
        If False, replace all column names with generic names like 'column_1', 'column_2', etc.
    rows_to_display : int, default 5
        Number of rows from the top of the DataFrame to show.
    """
    # Create a copy to avoid modifying the original
    temp_df = df.copy()

    # If using generic column names
    if not use_real_column_names:
        renamed_cols = {
            original_name: f"column_{i+1}" 
            for i, original_name in enumerate(temp_df.columns)
        }
        temp_df.rename(columns=renamed_cols, inplace=True)

    # Grab the top rows
    #head_df = temp_df.head(rows_to_display)
    head_df = temp_df.sample(rows_to_display, random_state=42)
    variations = {}

    # 1. Plain text (multi-line)
    variations["header_plain_text"] = head_df.to_string(index=False)

    # 2. Single-line format
    plain_text_lines = head_df.to_string(index=False).splitlines()
    variations["header_single_line"] = " | ".join(plain_text_lines)

    # 3. Minimal (column names only)
    variations["header_minimal"] = ", ".join(head_df.columns)

    # 4. CSV snippet
    variations["header_csv_snippet"] = head_df.to_csv(index=False)

    # 5. JSON snippet
    # (orient="records" -> list of objects; lines=False -> single JSON array)
    variations["header_json_snippet"] = head_df.to_json(orient="records", lines=False)

    # 6. Dict snippet (Python list of dicts)
    variations["header_dict_snippet"] = str(head_df.to_dict(orient="records"))

    # 7. Markdown table (requires tabulate)
    try:
        import tabulate
        variations["header_markdown_table"] = tabulate.tabulate(
            head_df, headers="keys", tablefmt="github", showindex=False
        )
    except ImportError:
        # Fallback if tabulate is not installed
        variations["header_markdown_table"] = head_df.to_string(index=False)

    return variations
def create_prompt_description(
    dataframe: pd.DataFrame,
    dataset_name: str = "",
    include_stats: bool = False,
    use_real_column_names: bool = True,
    rows_to_display: int = 5
) -> str:
    """
    Generates a prompt that instructs an LLM to create a structured data card
    (like a Kaggle dataset card) in JSON format.

    Parameters:
    -----------
    dataframe : pd.DataFrame                                                                                                                        The DataFrame for which to generate the prompt.
    dataset_name : str
        Name of the dataset.
    include_stats : bool, optional
        Whether to include basic descriptive statistics in the prompt.
    use_real_column_names : bool, optional
        If False, replace all column names with generic names like 'column_1', etc.
    rows_to_display : int, optional
        Number of rows from the top of the DataFrame to display in the preview.

    Returns:
    --------
    str
        A prompt string for an LLM that clearly specifies the desired JSON structure.
    """
    # Make a copy of the DataFrame for safe modifications
    preview_df = dataframe.copy()
    
    # Optionally replace column names with generic names
    if not use_real_column_names:
        renamed_cols = {old_name: f"column_{i+1}" for i, old_name in enumerate(preview_df.columns)}
        preview_df.rename(columns=renamed_cols, inplace=True)

    # Number of rows in the entire dataset
    total_rows = len(dataframe)

    # Get the preview of the DataFrame
    data_head_str = preview_df.head(rows_to_display).to_csv(index=False)

    # Optionally include descriptive statistics
    data_stats_str = ""
    if include_stats:
        stats_df = preview_df.describe(include="all").fillna("")  # Fill NaNs for display
        data_stats_str = "\n*Statistical Summary:*\n" + stats_df.to_string()

    # Create the final prompt
    prompt = f"""You are a data scientist helping to analyze the {dataset_name} dataset.
This dataset has a total of {total_rows} rows.

Below is a small preview of the data (first {rows_to_display} rows):
{data_head_str}
{data_stats_str}

**Your Task:**
1. **Columns Description**: For each column in the dataset, provide a brief explanation of what the column contains and how it might be used.
2. **Data Instance**: Provide a concise explanation of the records.
3. **Dataset Description**: Write a short, high-level overview of what this dataset represents and its purpose.
4. **Dataset Usage**: Describe potential use cases or scenarios for which this dataset could be applied.
5. **Title**: Provide a concise and clear title for the dataset.
6. **Fairness**: Identify any protected or sensitive variables in the dataset (e.g., race, gender, age). For each sensitive variable, specify:
   - Which groups might be privileged
   - Which groups might be unprivileged
   - Any potential biases arising from these variables

**Output Requirements:**

- **Return the response in JSON format**.
- The JSON must contain **exactly six keys**:
  1. "ColumnsDescription": object  
     - Each key should be a column name, and its value should be the column's description.
  2. "DataInstance": string  
  3. "DatasetDescription": string  
  4. "DatasetUsage": string  
  5. "Title": string  
  6. "Fairness": object  
     - Each key in this object should be the name of a protected/sensitive variable.  
     - The value for each key should be an object describing the privileged and unprivileged groups.

Do not provide any additional fields or commentary outside of these JSON keys.
"""
    return prompt.strip()

def load_model(model_name: str, prompt_text: str) -> str:
    """
    Load the specified model and generate a response based on the prompt_text.
    This function serves as a placeholder for different model integrations.
    
    Parameters
    ----------
    model_name : str
        Name of the model (e.g., 'GPT-4o', 'Mistral', 'Claude', 'OpenAI', etc.)
    prompt_text : str
        The prompt string to be passed to the model.

    Returns
    -------
    str
        The model's generated response as a string.
    """
    # Below are placeholders. You'd replace them with actual calls to your LLM APIs,
    # such as OpenAI, Anthropic (Claude), or Hugging Face for Mistral.

    if model_name == "GPT-4":
        client = OpenAI()
        response = client.chat.completions.create(model="gpt-4",messages=[{"role": "user", "content": prompt_text}],temperature=0)

        return response.choices[0].message.content

    elif model_name == "Mistral":
        import os
        from mistralai import Mistral
        
        api_key = os.environ["MISTRAL_API_KEY"]
        model = "mistral-large-latest"
        
        client = Mistral(api_key=api_key)

        chat_response = client.chat.complete(
            model = model,
            messages = [
                {
                    "role": "user",
                    "content": prompt_text},],temperature=0)

        return chat_response.choices[0].message.content

    elif model_name == "Claude":
        import anthropic
        client = anthropic.Anthropic(api_key="")
        message = client.messages.create(model="claude-3-5-sonnet-20241022",max_tokens=1024,messages=[{"role": "user", "content":prompt_text}],temperature=0)
        return message.content[0].text

    else:
        # If an unknown model is requested, return a warning or default
        return f"Model '{model_name}' not recognized."
