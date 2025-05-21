"""
Implementation of nodes for the LangGraph workflow.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.preprocessing import LabelEncoder

from utils import (
    load_dataset,
    analyze_class_distribution,
    detect_imbalance,
    visualize_distribution,
    save_visualization,
    prepare_data_for_resampling
)

def load_data_node(state: Dict) -> Dict:
    """
    Node for loading data from a CSV file.

    Args:
        state: Current state of the workflow

    Returns:
        Updated state with loaded data
    """
    file_path = state.get("file_path")
    if not file_path:
        return {
            "error": "No file path provided",
            "status": "failed"
        }

    try:
        df = load_dataset(file_path)

        return {
            "data": df,
            "columns": list(df.columns),
            "num_samples": len(df),
            "status": "success",
            "message": f"Successfully loaded dataset with {len(df)} samples and {len(df.columns)} columns"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

def analyze_distribution_node(state: Dict) -> Dict:
    """
    Node for analyzing class distribution.

    Args:
        state: Current state of the workflow

    Returns:
        Updated state with distribution analysis
    """
    df = state.get("data")
    target_column = state.get("target_column")

    if df is None:
        return {
            "error": "No data available",
            "status": "failed"
        }

    if target_column is None:
        return {
            "error": "No target column specified",
            "status": "failed"
        }

    try:
        distribution = analyze_class_distribution(df, target_column)

        # Create visualization
        fig = visualize_distribution(df, target_column, "Original Class Distribution")

        # Save visualization if output directory is provided
        output_dir = state.get("output_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_visualization(fig, os.path.join(output_dir, "original_distribution.png"))

        return {
            "distribution": distribution,
            "status": "success",
            "message": f"Successfully analyzed class distribution for {target_column}"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

def detect_imbalance_node(state: Dict) -> Dict:
    """
    Node for detecting class imbalance.

    Args:
        state: Current state of the workflow

    Returns:
        Updated state with imbalance detection results
    """
    distribution = state.get("distribution")

    if distribution is None:
        return {
            "error": "No distribution information available",
            "status": "failed"
        }

    try:
        imbalance_info = detect_imbalance(distribution)

        return {
            "imbalance_info": imbalance_info,
            "status": "success",
            "message": f"Imbalance detection completed. Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

def recommend_technique_node(state: Dict) -> Dict:
    """
    Node for recommending an appropriate resampling technique.
    Uses LLM to reason about the best technique based on the dataset characteristics.

    Args:
        state: Current state of the workflow

    Returns:
        Updated state with recommended technique
    """
    distribution = state.get("distribution")
    imbalance_info = state.get("imbalance_info")
    data = state.get("data")

    if distribution is None or imbalance_info is None or data is None:
        print("ERROR: Missing required information in recommend_technique_node")
        # Instead of failing, provide a default recommendation
        return {
            "recommendation": "Technique: SMOTE\nReason: Default technique selected due to missing information.",
            "recommended_technique": "SMOTE",
            "status": "success",
            "message": "Using default technique due to missing information: SMOTE"
        }

    try:
        # Create a prompt for the LLM
        prompt = ChatPromptTemplate.from_template("""
        You are an expert data scientist specializing in handling class imbalance problems.

        Dataset information:
        - Total samples: {total_samples}
        - Number of classes: {num_classes}
        - Class distribution: {class_counts}
        - Class percentages: {class_percentages}

        Imbalance information:
        - Is imbalanced: {is_imbalanced}
        - Imbalance ratio: {imbalance_ratio}
        - Minority class: {minority_class}
        - Majority class: {majority_class}
        - Severity: {severity}

        Based on this information, recommend the most appropriate resampling technique from the following options:
        1. SMOTE (Synthetic Minority Over-sampling Technique)
        2. Random Over-sampling
        3. Random Under-sampling
        4. NearMiss
        5. SMOTEENN (SMOTE + Edited Nearest Neighbors)
        6. SMOTETomek (SMOTE + Tomek links)

        Provide your recommendation in the following format:
        Technique: [name of technique]
        Reason: [detailed explanation of why this technique is appropriate]
        Parameters: [any specific parameters that should be set]
        """)

        try:
            # Get the LLM - Using Groq with Llama-4-Scout model
            print("Initializing Groq API client...")
            llm = ChatGroq(
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0,
                max_tokens=1000,
                groq_api_key=os.environ.get("GROQ_API_KEY")
            )

            # Format the prompt with the data
            print("Formatting prompt with dataset information...")
            formatted_prompt = prompt.format(
                total_samples=distribution["total_samples"],
                num_classes=distribution["num_classes"],
                class_counts=distribution["class_counts"],
                class_percentages=distribution["class_percentages"],
                is_imbalanced=imbalance_info["is_imbalanced"],
                imbalance_ratio=imbalance_info["imbalance_ratio"],
                minority_class=imbalance_info["minority_class"],
                majority_class=imbalance_info["majority_class"],
                severity=imbalance_info["severity"]
            )

            # Get the recommendation from the LLM
            print("Calling Groq API for technique recommendation...")
            response = llm.invoke(formatted_prompt)
            recommendation = response.content
            print(f"Received recommendation from Groq API: {recommendation[:100]}...")

            # Parse the recommendation
            technique_lines = [line for line in recommendation.split('\n') if line.startswith("Technique:")]

            if technique_lines:
                technique_line = technique_lines[0]
                technique = technique_line.split("Technique:")[1].strip()
                print(f"Successfully parsed technique: {technique}")
            else:
                # Fallback if the format is not as expected
                print("Warning: Could not parse technique from LLM response. Using default technique (SMOTE).")
                technique = "SMOTE"
                recommendation = f"Technique: {technique}\nReason: Default technique selected due to parsing error."

            return {
                "recommendation": recommendation,
                "recommended_technique": technique,
                "status": "success",
                "message": f"Successfully recommended technique: {technique}"
            }
        except Exception as llm_error:
            # Fallback to a default recommendation if the LLM call fails
            print(f"Error calling Groq API: {str(llm_error)}")
            print("Using fallback recommendation...")

            # Determine a reasonable default based on imbalance severity
            severity = imbalance_info.get("severity", "moderate")
            if severity == "severe":
                technique = "SMOTE"
                reason = "Default recommendation for severe imbalance when LLM call fails."
            elif severity == "moderate":
                technique = "Random Over-sampling"
                reason = "Default recommendation for moderate imbalance when LLM call fails."
            else:
                technique = "Random Under-sampling"
                reason = "Default recommendation for mild imbalance when LLM call fails."

            recommendation = f"Technique: {technique}\nReason: {reason}"
            print(f"Using fallback technique: {technique}")

            return {
                "recommendation": recommendation,
                "recommended_technique": technique,
                "status": "success",
                "message": f"Using fallback technique due to LLM error: {technique}"
            }
    except Exception as e:
        print(f"Error in recommend_technique_node: {str(e)}")
        # Provide a default recommendation even in case of general errors
        return {
            "recommendation": "Technique: SMOTE\nReason: Default fallback due to error.",
            "recommended_technique": "SMOTE",
            "status": "success",
            "message": f"Using default technique due to error: SMOTE. Error: {str(e)}"
        }

def apply_resampling_node(state: Dict) -> Dict:
    """
    Node for applying the recommended resampling technique.

    Args:
        state: Current state of the workflow

    Returns:
        Updated state with resampled data
    """
    data = state.get("data")
    target_column = state.get("target_column")
    recommended_technique = state.get("recommended_technique")

    if data is None or target_column is None or recommended_technique is None:
        return {
            "error": "Missing required information",
            "status": "failed"
        }

    try:
        # Prepare data for resampling
        X, y = prepare_data_for_resampling(data, target_column)

        # Apply the recommended technique
        resampling_techniques = {
            "SMOTE": SMOTE(random_state=42),
            "Random Over-sampling": RandomOverSampler(random_state=42),
            "Random Under-sampling": RandomUnderSampler(random_state=42),
            "NearMiss": NearMiss(version=1),
            "SMOTEENN": SMOTEENN(random_state=42),
            "SMOTETomek": SMOTETomek(random_state=42)
        }

        # Find the closest matching technique
        technique_key = None
        for key in resampling_techniques.keys():
            if key.lower() in recommended_technique.lower():
                technique_key = key
                break

        if technique_key is None:
            # Default to SMOTE if no match found
            technique_key = "SMOTE"

        # Apply resampling
        X_resampled, y_resampled = resampling_techniques[technique_key].fit_resample(X, y)

        # Convert back to DataFrame
        resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_df[target_column] = y_resampled

        return {
            "resampled_data": resampled_df,
            "applied_technique": technique_key,
            "original_shape": data.shape,
            "resampled_shape": resampled_df.shape,
            "status": "success",
            "message": f"Successfully applied {technique_key}. Original shape: {data.shape}, Resampled shape: {resampled_df.shape}"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

def visualize_results_node(state: Dict) -> Dict:
    """
    Node for visualizing the results of resampling.

    Args:
        state: Current state of the workflow

    Returns:
        Updated state with visualization paths
    """
    original_data = state.get("data")
    resampled_data = state.get("resampled_data")
    target_column = state.get("target_column")
    output_dir = state.get("output_dir")

    if original_data is None or resampled_data is None or target_column is None:
        return {
            "error": "Missing required information",
            "status": "failed"
        }

    try:
        # Create visualizations
        original_fig = visualize_distribution(original_data, target_column, "Original Class Distribution")
        resampled_fig = visualize_distribution(resampled_data, target_column, "Resampled Class Distribution")

        visualization_paths = {}

        # Save visualizations if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            original_path = os.path.join(output_dir, "original_distribution.png")
            resampled_path = os.path.join(output_dir, "resampled_distribution.png")

            save_visualization(original_fig, original_path)
            save_visualization(resampled_fig, resampled_path)

            visualization_paths = {
                "original": original_path,
                "resampled": resampled_path
            }

        return {
            "visualization_paths": visualization_paths,
            "status": "success",
            "message": "Successfully created visualizations"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

def save_results_node(state: Dict) -> Dict:
    """
    Node for saving the resampled dataset.

    Args:
        state: Current state of the workflow

    Returns:
        Updated state with saved file path
    """
    resampled_data = state.get("resampled_data")
    output_dir = state.get("output_dir")

    if resampled_data is None:
        return {
            "error": "No resampled data available",
            "status": "failed"
        }

    try:
        saved_path = None

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            saved_path = os.path.join(output_dir, "resampled_data.csv")
            resampled_data.to_csv(saved_path, index=False)

        return {
            "saved_path": saved_path,
            "status": "success",
            "message": f"Successfully saved resampled data to {saved_path}" if saved_path else "Resampled data not saved (no output directory specified)"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }
