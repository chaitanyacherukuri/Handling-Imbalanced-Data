"""
Utility functions for the class imbalance agent.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional
from sklearn.model_selection import train_test_split
from collections import Counter

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def analyze_class_distribution(df: pd.DataFrame, target_column: str) -> Dict:
    """
    Analyze the class distribution in the dataset.
    
    Args:
        df: DataFrame containing the dataset
        target_column: Name of the target column
        
    Returns:
        Dictionary with class distribution information
    """
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
    """
    Detect if there's a class imbalance in the dataset.
    
    Args:
        distribution: Dictionary with class distribution information
        
    Returns:
        Dictionary with imbalance detection results
    """
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
    """
    Visualize the class distribution.
    
    Args:
        df: DataFrame containing the dataset
        target_column: Name of the target column
        title: Title for the plot
        
    Returns:
        Matplotlib figure object
    """
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
    """
    Save a visualization to a file.
    
    Args:
        fig: Matplotlib figure object
        file_path: Path to save the figure
    """
    fig.savefig(file_path)
    plt.close(fig)

def prepare_data_for_resampling(df: pd.DataFrame, target_column: str, 
                               features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for resampling by separating features and target.
    
    Args:
        df: DataFrame containing the dataset
        target_column: Name of the target column
        features: List of feature columns (if None, all columns except target are used)
        
    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series
    """
    if features is None:
        features = [col for col in df.columns if col != target_column]
    
    X = df[features]
    y = df[target_column]
    
    return X, y
