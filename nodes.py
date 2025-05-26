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

def get_default_ml_algorithms(dataset_size: int, num_features: int) -> Dict:
    """Get default ML algorithm recommendations based on dataset characteristics."""
    algorithms = []

    if dataset_size > 50000:  # Large dataset
        algorithms = [
            {"name": "Random Forest", "reason": "Efficient for large datasets with good performance"},
            {"name": "Logistic Regression", "reason": "Fast training and prediction for large datasets"},
            {"name": "XGBoost", "reason": "Excellent performance on large tabular datasets"}
        ]
    elif dataset_size > 10000:  # Medium dataset
        algorithms = [
            {"name": "Random Forest", "reason": "Robust performance across various dataset sizes"},
            {"name": "SVM", "reason": "Good performance on medium-sized datasets"},
            {"name": "XGBoost", "reason": "Strong performance on structured data"}
        ]
    else:  # Small dataset
        algorithms = [
            {"name": "Random Forest", "reason": "Less prone to overfitting on small datasets"},
            {"name": "Logistic Regression", "reason": "Simple and interpretable for small datasets"},
            {"name": "Naive Bayes", "reason": "Works well with limited training data"}
        ]

    return {
        "algorithms": algorithms,
        "reasoning": f"Recommendations based on dataset size ({dataset_size} samples) and {num_features} features"
    }

@handle_node_errors
def recommend_ml_algorithm_node(state: Dict) -> Dict:
    """Recommend optimal ML classification algorithms for the resampled dataset."""
    resampled_data = state.get("resampled_data")
    applied_technique = state.get("applied_technique")
    target_column = state.get("target_column")

    if resampled_data is None:
        raise ValueError("No resampled data available")

    # Extract dataset characteristics
    dataset_size = len(resampled_data)
    num_features = len(resampled_data.columns) - 1  # Exclude target column

    # Analyze feature types
    feature_cols = [col for col in resampled_data.columns if col != target_column]
    numeric_features = len(resampled_data[feature_cols].select_dtypes(include=['number']).columns)
    categorical_features = num_features - numeric_features

    # Try LLM recommendation with fallback
    try:
        llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0,
            max_tokens=800,
            groq_api_key=os.environ.get("GROQ_API_KEY")
        )

        prompt = f"""Recommend 2-3 optimal machine learning classification algorithms for this resampled dataset:

Dataset Characteristics:
- Size: {dataset_size} samples
- Features: {num_features} total ({numeric_features} numeric, {categorical_features} categorical)
- Applied resampling: {applied_technique}
- Target column: {target_column}

Consider:
- Dataset size and computational efficiency
- Feature types and preprocessing applied
- Impact of resampling technique on algorithm choice
- Performance vs interpretability trade-offs

Choose from: Random Forest, SVM, Logistic Regression, XGBoost, Neural Networks, Naive Bayes

Format your response as:
Algorithm 1: [name]
Reason: [brief explanation]

Algorithm 2: [name]
Reason: [brief explanation]

Algorithm 3: [name]
Reason: [brief explanation]"""

        response = llm.invoke(prompt)
        recommendation_text = response.content

        # Parse LLM response with improved logic
        algorithms = []
        lines = recommendation_text.split('\n')
        current_algorithm = None
        in_negative_section = False

        for line in lines:
            line = line.strip()

            # Skip negative sections
            if "I did not choose" in line or "not recommended" in line.lower():
                in_negative_section = True
                continue

            if in_negative_section and line.startswith("*"):
                continue  # Skip bullet points in negative section

            if line.startswith("Algorithm") and ":" in line:
                in_negative_section = False
                current_algorithm = line.split(":", 1)[1].strip()
                # Clean up algorithm name
                for target_alg in ["Random Forest", "SVM", "Logistic Regression", "XGBoost", "Neural Networks", "Naive Bayes"]:
                    if target_alg in current_algorithm:
                        current_algorithm = target_alg
                        break
            elif line.startswith("Reason:") and current_algorithm:
                reason = line.split(":", 1)[1].strip()
                algorithms.append({"name": current_algorithm, "reason": reason})
                current_algorithm = None
            # Alternative parsing for **Algorithm: Name** format
            elif line.startswith("**Algorithm") and "**" in line:
                in_negative_section = False
                # Extract algorithm name from **Algorithm1: Random Forest** format
                if ":" in line:
                    alg_part = line.split(":", 1)[1].replace("**", "").strip()
                    for target_alg in ["Random Forest", "SVM", "Logistic Regression", "XGBoost", "Neural Networks", "Naive Bayes"]:
                        if target_alg in alg_part:
                            current_algorithm = target_alg
                            break

        # Fallback if parsing failed
        if not algorithms:
            print(f"Debug: LLM response parsing failed. Response was: {recommendation_text[:200]}...")
            fallback = get_default_ml_algorithms(dataset_size, num_features)
            algorithms = fallback["algorithms"]
            recommendation_text = f"LLM parsing failed. {fallback['reasoning']}"

        return {
            "ml_algorithm_recommendations": {
                "algorithms": algorithms,
                "recommendation_text": recommendation_text,
                "source": "LLM"
            },
            "message": f"LLM recommended {len(algorithms)} ML algorithms for the resampled dataset"
        }

    except Exception as e:
        # Fallback to default recommendations
        print(f"Info: LLM call failed ({str(e)[:50]}...), using fallback ML algorithm recommendations")
        fallback = get_default_ml_algorithms(dataset_size, num_features)

        return {
            "ml_algorithm_recommendations": {
                "algorithms": fallback["algorithms"],
                "recommendation_text": fallback["reasoning"],
                "source": "Fallback"
            },
            "message": f"Using fallback recommendations: {len(fallback['algorithms'])} ML algorithms suggested"
        }
