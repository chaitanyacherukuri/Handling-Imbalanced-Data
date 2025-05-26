"""
Implementation of nodes for the LangGraph workflow.
"""
import os
import pandas as pd
from typing import Dict
from langchain_groq import ChatGroq
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

from utils import (
    load_dataset,
    analyze_class_distribution,
    detect_imbalance,
    visualize_distribution,
    save_visualization,
    prepare_data_for_resampling,
    handle_node_errors,
    ensure_output_dir
)

@handle_node_errors
def load_data_node(state: Dict) -> Dict:
    """Node for loading data from a CSV file."""
    file_path = state.get("file_path")
    if not file_path:
        raise ValueError("No file path provided")

    df = load_dataset(file_path)
    return {
        "data": df,
        "columns": list(df.columns),
        "num_samples": len(df),
        "message": f"Successfully loaded dataset with {len(df)} samples and {len(df.columns)} columns"
    }

@handle_node_errors
def analyze_distribution_node(state: Dict) -> Dict:
    """Node for analyzing class distribution."""
    df = state.get("data")
    target_column = state.get("target_column")

    if df is None:
        raise ValueError("No data available")
    if target_column is None:
        raise ValueError("No target column specified")

    distribution = analyze_class_distribution(df, target_column)

    # Create and save visualization
    fig = visualize_distribution(df, target_column, "Original Class Distribution")
    output_dir = ensure_output_dir(state.get("output_dir"))
    if output_dir:
        save_visualization(fig, os.path.join(output_dir, "original_distribution.png"))

    return {
        "distribution": distribution,
        "message": f"Successfully analyzed class distribution for {target_column}"
    }

@handle_node_errors
def detect_imbalance_node(state: Dict) -> Dict:
    """Node for detecting class imbalance."""
    distribution = state.get("distribution")
    if distribution is None:
        raise ValueError("No distribution information available")

    imbalance_info = detect_imbalance(distribution)
    return {
        "imbalance_info": imbalance_info,
        "message": f"Imbalance detection completed. Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}"
    }

def get_default_technique(severity: str) -> str:
    """Get default technique based on imbalance severity."""
    return {
        "severe": "SMOTE",
        "moderate": "Random Over-sampling",
        "mild": "Random Under-sampling"
    }.get(severity, "SMOTE")

@handle_node_errors
def recommend_technique_node(state: Dict) -> Dict:
    """Node for recommending an appropriate resampling technique using LLM."""
    distribution = state.get("distribution")
    imbalance_info = state.get("imbalance_info")

    if not all([distribution, imbalance_info]):
        severity = imbalance_info.get("severity", "moderate") if imbalance_info else "moderate"
        technique = get_default_technique(severity)
        return {
            "recommendation": f"Technique: {technique}\nReason: Default technique for {severity} imbalance.",
            "recommended_technique": technique,
            "message": f"Using default technique: {technique}"
        }

    # Try LLM recommendation with simplified fallback
    try:
        llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0,
            max_tokens=500,
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )

        prompt = f"""Recommend a resampling technique for class imbalance:
- Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}
- Severity: {imbalance_info['severity']}
- Total samples: {distribution['total_samples']}

Choose from: SMOTE, Random Over-sampling, Random Under-sampling, NearMiss, SMOTEENN, SMOTETomek

Format: Technique: [name]
Reason: [brief explanation]"""

        response = llm.invoke(prompt)
        recommendation = response.content

        # Parse technique name
        for line in recommendation.split('\n'):
            if line.startswith("Technique:"):
                technique = line.split("Technique:")[1].strip()
                break
        else:
            technique = get_default_technique(imbalance_info["severity"])

        return {
            "recommendation": recommendation,
            "recommended_technique": technique,
            "message": f"LLM recommended technique: {technique}",
            "recommendation_source": "LLM"
        }

    except Exception as e:
        # Simple fallback without verbose logging
        technique = get_default_technique(imbalance_info["severity"])
        print(f"Info: LLM call failed ({str(e)[:50]}...), using fallback technique: {technique}")
        return {
            "recommendation": f"Technique: {technique}\nReason: Fallback recommendation for {imbalance_info['severity']} imbalance.",
            "recommended_technique": technique,
            "message": f"Using fallback technique: {technique}",
            "recommendation_source": "Fallback"
        }

@handle_node_errors
def apply_resampling_node(state: Dict) -> Dict:
    """Node for applying the recommended resampling technique."""
    data = state.get("data")
    target_column = state.get("target_column")
    recommended_technique = state.get("recommended_technique")

    if data is None or target_column is None or recommended_technique is None:
        raise ValueError("Missing required information")

    # Prepare data for resampling
    X, y = prepare_data_for_resampling(data, target_column)

    # Resampling techniques mapping
    techniques = {
        "SMOTE": SMOTE(random_state=42),
        "Random Over-sampling": RandomOverSampler(random_state=42),
        "Random Under-sampling": RandomUnderSampler(random_state=42),
        "NearMiss": NearMiss(version=1),
        "SMOTEENN": SMOTEENN(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42)
    }

    # Find matching technique or default to SMOTE
    technique_key = next(
        (key for key in techniques if key.lower() in recommended_technique.lower()),
        "SMOTE"
    )

    # Apply resampling
    X_resampled, y_resampled = techniques[technique_key].fit_resample(X, y)

    # Convert back to DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_df[target_column] = y_resampled

    return {
        "resampled_data": resampled_df,
        "applied_technique": technique_key,
        "original_shape": data.shape,
        "resampled_shape": resampled_df.shape,
        "message": f"Applied {technique_key}. Shape: {data.shape} → {resampled_df.shape}"
    }

@handle_node_errors
def visualize_results_node(state: Dict) -> Dict:
    """Node for visualizing the results of resampling."""
    original_data = state.get("data")
    resampled_data = state.get("resampled_data")
    target_column = state.get("target_column")

    if original_data is None or resampled_data is None or target_column is None:
        raise ValueError("Missing required information")

    # Create visualizations
    original_fig = visualize_distribution(original_data, target_column, "Original Class Distribution")
    resampled_fig = visualize_distribution(resampled_data, target_column, "Resampled Class Distribution")

    visualization_paths = {}
    output_dir = ensure_output_dir(state.get("output_dir"))

    if output_dir:
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
        "message": "Successfully created visualizations"
    }

@handle_node_errors
def save_results_node(state: Dict) -> Dict:
    """Node for saving the resampled dataset."""
    resampled_data = state.get("resampled_data")
    if resampled_data is None:
        raise ValueError("No resampled data available")

    saved_path = None
    output_dir = ensure_output_dir(state.get("output_dir"))

    if output_dir:
        saved_path = os.path.join(output_dir, "resampled_data.csv")
        resampled_data.to_csv(saved_path, index=False)

    return {
        "saved_path": saved_path,
        "message": f"Saved resampled data to {saved_path}" if saved_path else "Data not saved (no output directory)"
    }
