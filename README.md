# Class Imbalance Agent

An agentic AI application built with LangGraph to solve class imbalance problems in machine learning datasets.

## Overview

This application uses LangGraph to create an agentic workflow that:

1. Loads data from a CSV file
2. Analyzes class distribution
3. Detects class imbalance
4. Recommends an appropriate resampling technique using LLM reasoning
5. Applies the recommended technique
6. Visualizes the results
7. Saves the resampled dataset

## Project Structure

- `main.py`: Entry point for the application
- `agent.py`: Contains the LangGraph workflow definition
- `nodes.py`: Contains the implementation of each node in the workflow
- `utils.py`: Utility functions for data handling and visualization
- `data/`: Directory containing sample datasets
- `output/`: Directory where results are saved

## Requirements

- Python 3.8+
- LangGraph
- LangChain
- Groq API access with Llama-4-Scout model
- Pandas
- NumPy
- Matplotlib/Seaborn
- Scikit-learn
- Imbalanced-learn

## Installation

1. Clone the repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Set up Groq API key:

The application uses a .env file to load your Groq API key:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your credentials
# Replace the placeholder with your actual Groq API key
```

The .env file must contain the following variable:
- GROQ_API_KEY

You can get your Groq API key by signing up at https://console.groq.com/

The application will automatically load this API key when it runs.

## Usage

### Using a Sample Dataset

To generate and use a sample imbalanced dataset:

```bash
python main.py --generate-sample --output output --imbalance-ratio 10.0
```

This will:
- Generate a sample dataset with an imbalance ratio of 10:1
- Run the workflow on this dataset using Llama-4-Scout from Groq
- Save results to the `output` directory

### Using Your Own Dataset

To use your own dataset:

```bash
python main.py --file path/to/your/dataset.csv --target target_column_name --output output
```

Where:
- `path/to/your/dataset.csv` is the path to your CSV file
- `target_column_name` is the name of the target column in your dataset
- `output` is the directory where results will be saved

## Output

The application generates:
- Visualizations of the original and resampled class distributions
- The resampled dataset as a CSV file
- A JSON file with the full results of the workflow

## How It Works

1. **Data Loading**: The agent loads the dataset from a CSV file.
2. **Distribution Analysis**: It analyzes the class distribution in the dataset.
3. **Imbalance Detection**: It detects if there's a class imbalance and calculates metrics like imbalance ratio.
4. **Technique Recommendation**: Using Llama-4-Scout's reasoning capabilities through Groq, it recommends the most appropriate resampling technique based on the dataset characteristics.
5. **Resampling**: It applies the recommended technique to balance the dataset.
6. **Visualization**: It generates visualizations to compare the class distribution before and after resampling.
7. **Saving Results**: It saves the resampled dataset and visualizations.

## Groq and Llama-4-Scout

This application uses Meta's Llama-4-Scout model through Groq's API. Llama-4-Scout is a powerful open-source language model that offers:

- **Strong Reasoning**: Excellent reasoning capabilities for complex tasks like recommending resampling techniques
- **Fast Inference**: Groq's platform provides extremely fast inference speeds
- **Open Source Foundation**: Built on Meta's Llama 4 architecture, which is open source
- **Instruction Tuned**: Specifically tuned to follow instructions accurately

The application configures Llama-4-Scout with:
- Temperature set to 0 for deterministic outputs
- Maximum token limit of 1000 for responses

## Extending the Application

You can extend this application by:
- Adding more resampling techniques
- Implementing more sophisticated analysis
- Adding model training and evaluation after resampling
- Creating a web interface for the application

## License

MIT
