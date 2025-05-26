"""
Main entry point for the class imbalance agent.
"""
import os
import argparse
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv

from agent import run_workflow

def load_env_file():
    """Load environment variables from .env file."""
    if not os.path.exists('.env'):
        raise FileNotFoundError("Missing .env file with GROQ_API_KEY")

    load_dotenv()

    if not os.environ.get('GROQ_API_KEY'):
        raise ValueError("Missing GROQ_API_KEY in .env file")

def generate_sample_dataset(output_path: str, imbalance_ratio: float = 10.0) -> str:
    """Generate a sample imbalanced dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)

    # Generate imbalanced data
    n_majority, n_minority = 1000, int(1000 / imbalance_ratio)
    X = np.vstack([
        np.random.randn(n_majority, 5),
        np.random.randn(n_minority, 5)
    ])
    y = np.hstack([np.zeros(n_majority), np.ones(n_minority)])

    # Create and save DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["target"] = y.astype(int)
    df.to_csv(output_path, index=False)

    print(f"Generated dataset: {len(df)} samples, ratio {imbalance_ratio}:1")
    return output_path

def print_results(result: dict):
    """Print workflow results in a concise format."""
    # Get status from the final node result or overall result
    status = result.get("status", "unknown")
    if isinstance(result, dict) and len(result) == 1:
        # If result contains only one key (the final node), get status from there
        final_node_result = list(result.values())[0]
        if isinstance(final_node_result, dict):
            status = final_node_result.get("status", status)

    print(f"\nWorkflow Status: {status}")

    if status in ["success", "warning"]:
        # Extract data from nested result structure
        data = result
        if isinstance(result, dict) and len(result) == 1:
            data = list(result.values())[0]

        if info := data.get("imbalance_info"):
            print(f"Imbalance: {info.get('severity')} (ratio: {info.get('imbalance_ratio', 0):.2f})")

        if technique := data.get("applied_technique"):
            print(f"Applied: {technique}")
            print(f"Shape: {data.get('original_shape')} → {data.get('resampled_shape')}")

        if paths := data.get("visualization_paths"):
            print(f"Visualizations: {len(paths)} files created")

        if saved := data.get("saved_path"):
            print(f"Saved: {saved}")

        if ml_recs := data.get("ml_algorithm_recommendations"):
            algorithms = ml_recs.get("algorithms", [])
            print(f"ML Algorithms: {len(algorithms)} recommended")
            for i, alg in enumerate(algorithms, 1):
                print(f"  {i}. {alg.get('name', 'Unknown')} - {alg.get('reason', 'No reason provided')}")
    else:
        error_msg = result.get("error", "Unknown error")
        if isinstance(result, dict) and len(result) == 1:
            final_node_result = list(result.values())[0]
            if isinstance(final_node_result, dict):
                error_msg = final_node_result.get("error", error_msg)
        print(f"Error: {error_msg}")

def main():
    """Main function to run the class imbalance agent."""
    parser = argparse.ArgumentParser(description="Class Imbalance Agent")
    parser.add_argument("--file", type=str, help="Path to the CSV file")
    parser.add_argument("--target", type=str, help="Name of the target column")
    parser.add_argument("--output", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--generate-sample", action="store_true", help="Generate a sample dataset")
    parser.add_argument("--imbalance-ratio", type=float, default=10.0, help="Imbalance ratio")

    args = parser.parse_args()
    load_env_file()
    os.makedirs(args.output, exist_ok=True)

    # Determine input file and target
    if args.generate_sample:
        file_path = generate_sample_dataset(
            os.path.join(args.output, "sample_dataset.csv"),
            args.imbalance_ratio
        )
        target_column = "target"
    else:
        if not args.file or not args.target:
            parser.error("--file and --target are required when not using --generate-sample")
        file_path, target_column = args.file, args.target

    # Run workflow
    print(f"Processing: {file_path} (target: {target_column})")
    result = run_workflow(file_path, target_column, args.output)

    # Print results and save JSON
    print_results(result)

    # Save simplified result with better error handling
    result_path = os.path.join(args.output, "result.json")

    def make_serializable(obj):
        """Recursively make objects JSON serializable."""
        if hasattr(obj, 'shape'):  # DataFrame or numpy array
            return f"DataFrame/Array with shape {obj.shape}"
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    try:
        serializable_result = make_serializable(result) if result else {}
        with open(result_path, "w") as f:
            json.dump(serializable_result, f, indent=2)
        print(f"Results saved to: {result_path}")
    except Exception as e:
        print(f"Warning: Could not save results to JSON: {e}")
        print(f"Result type: {type(result)}, keys: {list(result.keys()) if result else 'None'}")

if __name__ == "__main__":
    main()
