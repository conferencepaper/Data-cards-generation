import json
import streamlit as st
import pandas as pd
import altair as alt
import os
from utils.Stats import plot_histogram_alphanumeric
from utils.Prompt_engineering import create_prompt_description, load_model, generate_fairness_prompt
from utils.Save_models import log_study_runV3
from utils.Prompt_engineeringV1 import create_prompt_description_target
from ast import literal_eval
import os


open_ai_key=""
os.environ["OPENAI_API_KEY"] = open_ai_key
#https://console.mistral.ai/api-keys/
mistral_ai_key=""
os.environ["MISTRAL_API_KEY"] = mistral_ai_key
claude_key=""
os.environ["ANTHROPIC_API_KEY"] = claude_key  # Replace with your actual Anthropic API key

def clean_metadata(metadata: dict) -> dict:
    cleaned_meta = {}
    for key, value in metadata.items():
        if isinstance(value, st.runtime.uploaded_file_manager.UploadedFile):
            cleaned_meta[key] = value.name  # store filename only
        else:
            cleaned_meta[key] = value
    return cleaned_meta

# 1. Initialize session_state variables so they persist across reruns
if "model_response" not in st.session_state:
    st.session_state["model_response"] = []

if "json_data" not in st.session_state:
    st.session_state["json_data"] = []

if "df" not in st.session_state:
    st.session_state["df"] = None

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

# ---------------------------------------------------------------------
# STREAMLIT TITLE
# ---------------------------------------------------------------------
st.title("Data Cards Generator")
use_real_column_names=True

# ---------------------------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

#metadata_file = st.file_uploader("Upload a JSON file", type=["json"])
metadata_file=None
if metadata_file:
    metadata = json.load(metadata_file)
else:
    metadata = {}

import pandas as pd
pd.set_option("styler.render.max_elements", 732630)

# ---------------------------------------------------------------------
# LOAD AND DISPLAY DATASET
# ---------------------------------------------------------------------
if uploaded_file is not None:
    if st.session_state["df"] is None:
        # Only load data once; store in session_state
        st.session_state["df"] = pd.read_csv(uploaded_file)
        st.session_state["dataset_name"] = uploaded_file.name

    df = st.session_state["df"]
    dataset_name = st.session_state["dataset_name"]

    st.subheader("Data Table")
    st.dataframe(df)

    # -------------------------------------------------------------
    # Generate Data Cards
    # -------------------------------------------------------------
    st.subheader("Generate Prompt-Based Data Cards")
    include_stats = st.checkbox("Include basic statistics?", value=False)
    use_real_column_names = st.checkbox("Use real column names?", value=True)
    model_name = st.selectbox(
        "Select a model:",
        ["GPT-4", "Claude", "Mistral", "OpenAI"]
    )
    rows_to_display = st.slider("Rows to display in prompt:", 1, 10, 5)

    # Generate Data Cards button
    if st.button("Generate Data Cards"):
        header_variation = "header_plain_text"
        prompt_text = create_prompt_description_target(
            dataframe=df,
            dataset_name=dataset_name,
            include_stats=include_stats,
            use_real_column_names=use_real_column_names,
            rows_to_display=rows_to_display
        )

        prompt_name = (
            f"{header_variation}"
            f"_stats-{include_stats}"
            f"_colnames-{use_real_column_names}"
            f"_rows-{rows_to_display}"
        )

        st.write("### Generated Prompt")
        st.write(prompt_text)

        response = load_model(model_name, prompt_text)
        if model_name=="Mistral":
            response=response[8:-4]
        st.write("### Model Response")
        st.json(response)  # Render as JSON if valid

        # Try to parse the response as JSON
        try:
            file_json = json.loads(response)
        except json.JSONDecodeError:
            st.write("Response is not valid JSON. Showing raw response:")
            # Optionally, adjust the slicing if needed
            raw_response = response[8:-4]
            st.write(raw_response)
            file_json = json.loads(raw_response)

        json_data = json.dumps(file_json, indent=2)

        # Log run (assuming clean_metadata and log_study_runV3 are defined)
        log_study_runV3(
            output_file="model_history_file.json",
            dataset_name=dataset_name,
            model_name=model_name,
            prompt_name=prompt_name,
            prompt=prompt_text,
            response=file_json,
            metadata=clean_metadata(metadata)
        )

        # Save for session history if needed
        st.session_state.setdefault("model_response", []).append(response)
        st.session_state.setdefault("json_data", []).append(json_data)

        ## Row 1: Header
        row1_left, row_middle, row1_right = st.columns(3)

        # Define a function that highlights columns if they are in the fairness list
        def highlight_fair_cols(s):
            return ['background-color: lightgreen' if s.name in fair_columns else '' for _ in s]
        
        fairness_info = file_json.get("Fairness", {})
        fair_columns = list(fairness_info.keys()) if fairness_info else []
        
        # Apply the styling and display the dataframe
        styled_df = df.style.apply(highlight_fair_cols, axis=0)
        
        with row_middle:
            st.subheader("Data Table")
            st.dataframe(styled_df)
        
        with row1_left:
            # Display Title, Dataset Description, Dataset Usage, and Data Instance
            title = file_json.get("Title", "No Title Provided")
            st.markdown("**Title:**")
            st.write(title)
            st.markdown("**Dataset Description:**")
            st.write(file_json.get("DatasetDescription", ""))
        
        with row1_right:
            st.markdown("**Dataset Usage:**")
            st.write(file_json.get("DatasetUsage", ""))
            st.markdown("**Data Instance:**")
            st.write(file_json.get("DataInstance", ""))

        ## Row 2: Columns Description and Fairness Information
        row2_left, row2_right = st.columns(2)

        with row2_left:
            st.markdown("**Columns Description:**")
            columns_desc = file_json.get("ColumnsDescription", {})
            if columns_desc:
                for col_name, description in columns_desc.items():
                    st.markdown(f"- **{col_name}**: {description}")
            else:
                st.write("No column descriptions provided.")

        with row2_right:
            st.markdown("**Fairness:**")
            fairness_info = file_json.get("Fairness", {})
            if fairness_info:
                # Check if fairness_info is a dictionary with nested items
                if isinstance(fairness_info, dict):
                    # Handle the case where Fairness is a flat dictionary with keys like "favorable_labels", "protected_attributes"
                    if "favorable_labels" in fairness_info:
                        st.markdown("**Favorable Labels:**")
                        favorable_labels = fairness_info.get("favorable_labels", [])
                        if isinstance(favorable_labels, list):
                            for label in favorable_labels:
                                st.markdown(f"  - {label}")
                        else:
                            st.markdown(f"  - {favorable_labels}")
                        
                    if "protected_attributes" in fairness_info:
                        st.markdown("**Protected Attributes:**")
                        protected_attrs = fairness_info.get("protected_attributes", [])
                        if isinstance(protected_attrs, list):
                            for attr in protected_attrs:
                                st.markdown(f"  - {attr}")
                        else:
                            st.markdown(f"  - {protected_attrs}")
                    
                    # For other potential fairness details not explicitly handled
                    for key, value in fairness_info.items():
                        if key not in ["favorable_labels", "protected_attributes"]:
                            st.markdown(f"**{key}:**")
                            if isinstance(value, dict):
                                # This is for nested structures like in your original code
                                privileged = value.get("privileged", [])
                                unprivileged = value.get("unprivileged", [])
                                potential_bias = value.get("potential_biases", "")
                                st.markdown(f"  - Privileged Groups: {privileged}")
                                st.markdown(f"  - Unprivileged Groups: {unprivileged}")
                                st.markdown(f"  - Potential Bias: {potential_bias}")
                            else:
                                st.markdown(f"  - {value}")
                
                # Handle if fairness_info is a list of dictionaries (like your original code expected)
                elif isinstance(fairness_info, list):
                    for fair in fairness_info:
                        if isinstance(fair, dict):
                            for variable, details in fair.items():
                                st.markdown(f"**{variable}:**")
                                privileged = details.get("privileged", [])
                                unprivileged = details.get("unprivileged", [])
                                potential_bias = details.get("potential_biases", "")
                                st.markdown(f"  - Privileged Groups: {privileged}")
                                st.markdown(f"  - Unprivileged Groups: {unprivileged}")
                                st.markdown(f"  - Potential Bias: {potential_bias}")
                        else:
                            st.markdown(f"  - {fair}")
                else:
                    # If fairness_info is a string or other non-iterable type
                    st.markdown(f"  - {fairness_info}")
            else:
                st.write("No fairness information provided.")
                
            st.markdown("**Rule Definition:**")
            rule_def = file_json.get("Rule_def", {})
            if rule_def:
                # Display Functional Dependencies
                st.markdown("### Functional Dependencies")
                func_deps = rule_def.get("FunctionalDependencies", [])
                if func_deps and len(func_deps) > 0:
                    for i, dep in enumerate(func_deps):
                        determinant = dep.get("determinant", [])
                        dependent = dep.get("dependent", [])
                        if determinant and dependent:
                            st.markdown(f"- **Dependency {i+1}**: {', '.join(determinant)} → {', '.join(dependent)}")
                else:
                    st.write("No functional dependencies identified.")
                
                # Display Conditional Functional Dependencies
                st.markdown("### Conditional Functional Dependencies")
                cond_deps = rule_def.get("ConditionalFunctionalDependencies", [])
                if cond_deps and len(cond_deps) > 0:
                    for i, dep in enumerate(cond_deps):
                        determinant = dep.get("determinant", [])
                        dependent = dep.get("dependent", [])
                        condition = dep.get("condition", "")
                        if determinant and dependent:
                            st.markdown(f"- **Dependency {i+1}**: {', '.join(determinant)} → {', '.join(dependent)} when {condition}")
                else:
                    st.write("No conditional functional dependencies identified.")
                
                # Display Denial Constraints
                st.markdown("### Denial Constraints")
                denial_constraints = rule_def.get("DenialConstraints", [])
                if denial_constraints and len(denial_constraints) > 0:
                    for i, constraint in enumerate(denial_constraints):
                        constraint_text = constraint.get("constraint", "")
                        if constraint_text:
                            st.markdown(f"- **Constraint {i+1}**: {constraint_text}")
                else:
                    st.write("No denial constraints identified.")
                
                # Display Other Constraints
                st.markdown("### Other Constraints")
                other_constraints = rule_def.get("OtherConstraints", {})
                if other_constraints and len(other_constraints) > 0:
                    if isinstance(other_constraints, list):
                        for i, constraint in enumerate(other_constraints):
                            st.markdown(f"- **Constraint {i+1}**: {constraint}")
                    else:
                        for key, value in other_constraints.items():
                            st.markdown(f"- **{key}**: {value}")
                else:
                    st.write("No other constraints identified.")
            else:
                st.write("No rule definition information provided.")

        # Download button for the JSON file
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"{uploaded_file.name}_generated_data_cards_{model_name}.json",
            mime="application/json"
        )

    # -------------------------------------------------------------
    # Fairness Analysis Section
    # -------------------------------------------------------------
    st.subheader("Fairness Analysis")
    # Plot histogram
    column_names = df.columns.tolist()
    selected_column = st.selectbox("Select a column to plot:", column_names)
    bins = st.slider("Number of bins:", min_value=1, max_value=50, value=10)
    if st.button("Plot Histogram"):
        fig = plot_histogram_alphanumeric(df, selected_column, bins=bins)
        st.pyplot(fig)

    # Here we select the target column, but do NOT reload entire data
    target_column_selected = st.selectbox("Select a Target column:", column_names, key="target_column_key")

    # Each time target_column_selected changes, we can generate a new fairness prompt
    prompt_fairness = generate_fairness_prompt(
        dataframe=df,
        dataset_name=dataset_name,
        include_stats=include_stats,
        header_variation="header_plain_text",
        rows_to_display=rows_to_display,
        target_columns=target_column_selected
    )

    st.write("### Generated Fairness Prompt")
    st.write(prompt_fairness)

    # A separate button to generate only the fairness analysis
    if st.button("Generate Fairness Analysis"):
        response_fairness = load_model(model_name, prompt_fairness)
        st.write("### Fairness Analysis Response")
        st.json(response_fairness)

        fairness_prompt_name = (
            f"fairness_{model_name}_target-{target_column_selected}"
        )

        # Log fairness run
        log_study_runV3(
            output_file="model_history_fairness_file.json",
            dataset_name=dataset_name,
            model_name=model_name,
            prompt_name=fairness_prompt_name,
            prompt=prompt_fairness,
            response=response_fairness,
            metadata=clean_metadata(metadata)
        )
