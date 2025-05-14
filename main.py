"""
Main entry point for the class imbalance agent.
"""
import os
import argparse
import pandas as pd
import numpy as np
from typing import Optional
import json
import sys
from dotenv import load_dotenv

from agent import run_workflow

def load_env_file(env_file='.env'):
    """
    Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file (default: '.env')
    """
    # Check if .env file exists
    if not os.path.exists(env_file):
        print(f"Warning: {env_file} file not found.")
        print("Please create a .env file with your Groq API key.")
        print("You can copy .env.example to .env and fill in your credentials.")
        sys.exit(1)

    # Load environment variables from .env file
    load_dotenv(env_file)

    # Check if required Groq API key is set
    if not os.environ.get('GROQ_API_KEY'):
        print("Error: Missing GROQ_API_KEY in .env file")
        print("Please make sure your .env file contains your Groq API key.")
        sys.exit(1)

    print("Groq API key loaded successfully from .env file.")

def generate_sample_dataset(output_path: str, imbalance_ratio: float = 10.0) -> str:
    """
    Generate a sample imbalanced dataset for testing.

    Args:
        output_path: Path to save the dataset
        imbalance_ratio: Ratio of majority to minority class

    Returns:
        Path to the generated dataset
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate features
    n_samples_majority = 1000
    n_samples_minority = int(n_samples_majority / imbalance_ratio)
    n_features = 5

    # Generate majority class samples
    X_majority = np.random.randn(n_samples_majority, n_features)
    y_majority = np.zeros(n_samples_majority)

    # Generate minority class samples
    X_minority = np.random.randn(n_samples_minority, n_features)
    y_minority = np.ones(n_samples_minority)

    # Combine the data
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])

    # Create a DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y.astype(int)

    # Save the dataset
    df.to_csv(output_path, index=False)

    print(f"Generated sample dataset with {len(df)} samples")
    print(f"Class distribution: {df['target'].value_counts().to_dict()}")

    return output_path

def main():
    """Main function to run the class imbalance agent."""
    parser = argparse.ArgumentParser(description="Class Imbalance Agent")
    parser.add_argument("--file", type=str, help="Path to the CSV file")
    parser.add_argument("--target", type=str, help="Name of the target column")
    parser.add_argument("--output", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a sample dataset")
    parser.add_argument("--imbalance-ratio", type=float, default=10.0,
                        help="Imbalance ratio for the sample dataset (majority:minority)")

    args = parser.parse_args()

    # Load Groq API key from .env file
    load_env_file()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate sample dataset if requested
    if args.generate_sample:
        sample_path = os.path.join(args.output, "sample_dataset.csv")
        file_path = generate_sample_dataset(sample_path, args.imbalance_ratio)
        target_column = "target"
    else:
        # Use provided file and target column
        if not args.file or not args.target:
            parser.error("--file and --target are required when not using --generate-sample")

        file_path = args.file
        target_column = args.target

    # Run the workflow
    print(f"Running workflow with file: {file_path}, target: {target_column}")
    result = run_workflow(file_path, target_column, args.output)

    # Print the final result
    print("\n" + "="*50)
    print("Workflow completed")
    print("="*50)

    # Print key results
    if result.get("status") == "success":
        print(f"Status: {result.get('status')}")
        print(f"Message: {result.get('message')}")

        if result.get("imbalance_info"):
            print("\nImbalance Information:")
            imbalance_info = result.get("imbalance_info")
            print(f"Is imbalanced: {imbalance_info.get('is_imbalanced')}")
            print(f"Imbalance ratio: {imbalance_info.get('imbalance_ratio'):.2f}")
            print(f"Severity: {imbalance_info.get('severity')}")

        if result.get("recommended_technique"):
            print(f"\nRecommended technique: {result.get('recommended_technique')}")

        if result.get("applied_technique"):
            print(f"\nApplied technique: {result.get('applied_technique')}")
            print(f"Original shape: {result.get('original_shape')}")
            print(f"Resampled shape: {result.get('resampled_shape')}")

        if result.get("visualization_paths"):
            print("\nVisualization paths:")
            for name, path in result.get("visualization_paths").items():
                print(f"- {name}: {path}")

        if result.get("saved_path"):
            print(f"\nResampled data saved to: {result.get('saved_path')}")
    else:
        print(f"Status: {result.get('status')}")
        print(f"Error: {result.get('error')}")

    # Save the full result as JSON
    result_path = os.path.join(args.output, "result.json")

    # Convert DataFrame objects to strings for JSON serialization
    serializable_result = {}
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            serializable_result[key] = f"DataFrame with shape {value.shape}"
        else:
            serializable_result[key] = value

    with open(result_path, "w") as f:
        json.dump(serializable_result, f, indent=2)

    print(f"\nFull result saved to: {result_path}")

if __name__ == "__main__":
    main()
