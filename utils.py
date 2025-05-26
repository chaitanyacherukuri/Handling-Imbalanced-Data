"""
Utility functions for the class imbalance agent.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional, List, Callable, Any
from functools import wraps
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def handle_node_errors(func: Callable) -> Callable:
    """Decorator to handle errors in workflow nodes."""
    @wraps(func)
    def wrapper(state: Dict) -> Dict:
        try:
            result = func(state)
            if "status" not in result:
                result["status"] = "success"
            return {**state, **result}
        except Exception as e:
            # Continue workflow with error info but don't fail
            print(f"Warning in {func.__name__}: {str(e)}")
            return {**state, "error": str(e), "status": "warning", "message": f"Node {func.__name__} had an error: {str(e)}"}
    return wrapper

def ensure_output_dir(output_dir: Optional[str]) -> Optional[str]:
    """Ensure output directory exists."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def analyze_class_distribution(df: pd.DataFrame, target_column: str) -> Dict:
    """Analyze the class distribution in the dataset."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    class_counts = df[target_column].value_counts().to_dict()
    total_samples = len(df)
    class_percentages = {cls: (count / total_samples) * 100
                         for cls, count in class_counts.items()}

    return {
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "total_samples": total_samples,
        "num_classes": len(class_counts)
    }

def detect_imbalance(distribution: Dict) -> Dict:
    """Detect if there's a class imbalance in the dataset."""
    class_percentages = distribution["class_percentages"]
    min_class = min(class_percentages.items(), key=lambda x: x[1])
    max_class = max(class_percentages.items(), key=lambda x: x[1])
    imbalance_ratio = max_class[1] / min_class[1] if min_class[1] > 0 else float('inf')

    # Determine imbalance severity
    if imbalance_ratio > 10:
        severity = "severe"
    elif imbalance_ratio > 3:
        severity = "moderate"
    else:
        severity = "mild"

    is_imbalanced = imbalance_ratio > 1.5  # Threshold for considering imbalanced

    return {
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
        "minority_class": min_class[0],
        "majority_class": max_class[0],
        "severity": severity
    }

def visualize_distribution(df: pd.DataFrame, target_column: str, title: str = "Class Distribution") -> plt.Figure:
    """Visualize the class distribution."""
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=target_column, data=df)

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom')

    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()

    return plt.gcf()

def save_visualization(fig: plt.Figure, file_path: str) -> None:
    """Save a visualization to a file."""
    fig.savefig(file_path)
    plt.close(fig)

def prepare_data_for_resampling(df: pd.DataFrame, target_column: str,
                               features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for resampling with advanced preprocessing including imputation and encoding."""
    print(f"Starting data preprocessing for resampling...")
    print(f"Original dataset shape: {df.shape}")

    if features is None:
        features = [col for col in df.columns if col != target_column]

    X = df[features].copy()
    y = df[target_column].copy()

    # Handle target variable missing values first
    target_missing = y.isnull().sum()
    if target_missing > 0:
        print(f"Warning: Removing {target_missing} rows with missing target values")
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]

    print(f"After target cleaning: {X.shape[0]} samples")

    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")

    processed_features = []

    # Process numeric features with imputation
    if numeric_cols:
        print(f"Processing numeric features: {numeric_cols}")
        X_numeric = X[numeric_cols].copy()

        # Check for missing values in numeric columns
        numeric_missing = X_numeric.isnull().sum()
        cols_with_missing = numeric_missing[numeric_missing > 0]

        if len(cols_with_missing) > 0:
            print(f"Imputing missing values in {len(cols_with_missing)} numeric columns:")
            for col, missing_count in cols_with_missing.items():
                pct_missing = (missing_count / len(X_numeric)) * 100
                print(f"  - {col}: {missing_count} missing ({pct_missing:.1f}%)")

            # Use KNN imputation for better quality, fallback to median for large datasets
            if len(X_numeric) > 10000:
                print("Using median imputation for large dataset")
                imputer = SimpleImputer(strategy='median')
            else:
                print("Using KNN imputation for better quality")
                imputer = KNNImputer(n_neighbors=5)

            X_numeric_imputed = pd.DataFrame(
                imputer.fit_transform(X_numeric),
                columns=X_numeric.columns,
                index=X_numeric.index
            )
        else:
            print("No missing values in numeric features")
            X_numeric_imputed = X_numeric

        processed_features.append(X_numeric_imputed)

    # Process categorical features with encoding and imputation
    if categorical_cols:
        print(f"Processing categorical features: {categorical_cols}")
        X_categorical = X[categorical_cols].copy()

        categorical_encoded_dfs = []

        for col in categorical_cols:
            col_data = X_categorical[col].copy()
            missing_count = col_data.isnull().sum()

            if missing_count > 0:
                pct_missing = (missing_count / len(col_data)) * 100
                print(f"  - {col}: {missing_count} missing ({pct_missing:.1f}%) - imputing with mode")
                # Impute categorical missing values with mode
                mode_value = col_data.mode().iloc[0] if not col_data.mode().empty else 'Unknown'
                col_data = col_data.fillna(mode_value)

            # Encode categorical variables
            unique_values = col_data.nunique()
            print(f"  - {col}: {unique_values} unique values - using label encoding")

            # Use label encoding for categorical variables
            le = LabelEncoder()
            col_encoded = pd.DataFrame({
                f"{col}_encoded": le.fit_transform(col_data.astype(str))
            }, index=col_data.index)

            categorical_encoded_dfs.append(col_encoded)

        if categorical_encoded_dfs:
            X_categorical_encoded = pd.concat(categorical_encoded_dfs, axis=1)
            processed_features.append(X_categorical_encoded)

    # Combine all processed features
    if not processed_features:
        raise ValueError("No features available after preprocessing")

    X_processed = pd.concat(processed_features, axis=1)

    # Final validation
    remaining_missing = X_processed.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain after preprocessing")
        # Final cleanup - remove any remaining missing values
        mask = ~X_processed.isnull().any(axis=1)
        X_processed = X_processed[mask]
        y = y[mask]
        print(f"Removed {(~mask).sum()} rows with remaining missing values")

    print(f"Final preprocessed dataset shape: {X_processed.shape}")
    print(f"Data retention: {len(X_processed)}/{df.shape[0]} samples ({(len(X_processed)/df.shape[0]*100):.1f}%)")
    print(f"Features after preprocessing: {list(X_processed.columns)}")

    return X_processed, y
