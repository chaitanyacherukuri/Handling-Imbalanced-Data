# 🚀 LangGraph Data Imbalance Handler

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.19%2B-green.svg)](https://github.com/langchain-ai/langgraph)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange.svg)](https://aws.amazon.com/bedrock/)
[![Claude](https://img.shields.io/badge/Claude-3.7%20Sonnet-purple.svg)](https://www.anthropic.com/claude)

An intelligent agentic AI application built with LangGraph that provides comprehensive data imbalance handling with ML algorithm recommendations powered by Claude 3.7 Sonnet via AWS Bedrock.

## 📊 Overview

This application creates an intelligent LangGraph workflow that automatically:

1. **📁 Loads data** from CSV files with advanced preprocessing
2. **📈 Analyzes class distribution** with detailed metrics
3. **🔍 Detects class imbalance** using configurable thresholds
4. **🤖 Recommends resampling techniques** using LLM reasoning
5. **⚖️ Applies intelligent resampling** with multiple algorithms
6. **📊 Visualizes results** with before/after comparisons
7. **💾 Saves resampled datasets** in multiple formats
8. **🎯 Recommends ML algorithms** optimized for the resampled data

## 🏗️ Workflow Architecture

The application implements an 8-node LangGraph workflow with intelligent error handling and state management:

| Node | Function | Description |
|------|----------|-------------|
| 🔄 **load_data** | Data ingestion | Loads CSV files with validation and error handling |
| 📊 **analyze_distribution** | Statistical analysis | Computes class distribution metrics and percentages |
| 🔍 **detect_imbalance** | Imbalance detection | Calculates imbalance ratios and severity classification |
| 🤖 **recommend_technique** | LLM-powered recommendations | Uses Claude 3.7 Sonnet to suggest optimal resampling techniques |
| ⚖️ **apply_resampling** | Data resampling | Applies recommended techniques with advanced preprocessing |
| 📈 **visualize_results** | Visualization generation | Creates before/after distribution plots |
| 💾 **save_results** | Data persistence | Saves resampled datasets and metadata |
| 🎯 **recommend_ml_algorithm** | ML algorithm suggestions | Provides intelligent algorithm recommendations |

### Workflow Sequence
```
load_data → analyze_distribution → detect_imbalance → recommend_technique →
apply_resampling → visualize_results → save_results → recommend_ml_algorithm → END
```

## 📁 Project Structure

```
├── main.py              # Entry point with CLI interface
├── agent.py             # LangGraph workflow definition and state management
├── nodes.py             # Implementation of all 8 workflow nodes
├── utils.py             # Advanced preprocessing and utility functions
├── requirements.txt     # Python dependencies with version pinning
├── .env                 # Environment variables (AWS credentials)
├── data/               # Sample datasets and input files
└── output/             # Generated results, visualizations, and reports
```

## ⚙️ Requirements & Dependencies

### Core Dependencies
```
langgraph>=0.0.19        # Workflow orchestration framework
langchain>=0.0.335       # LLM integration and chains
langchain-aws>=0.2.6     # AWS Bedrock integration for Claude 3.7 Sonnet
boto3>=1.34.84           # AWS SDK for Python
pandas>=2.0.0            # Data manipulation and analysis
numpy>=1.24.0            # Numerical computing
matplotlib>=3.7.0        # Plotting and visualization
seaborn>=0.12.0          # Statistical data visualization
scikit-learn>=1.2.0      # Machine learning utilities
imbalanced-learn>=0.10.0 # Specialized resampling algorithms
```

### System Requirements
- **Python**: 3.8+ (recommended: 3.9+)
- **Memory**: 4GB+ RAM (8GB+ for large datasets)
- **Storage**: 1GB+ free space for outputs
- **AWS Access**: AWS credentials with Bedrock permissions for Claude 3.7 Sonnet

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/chaitanyacherukuri/Handling-Imbalanced-Data.git
cd Handling-Imbalanced-Data
```

### 2. Install Dependencies
```bash
# Using pip (recommended)
python -m pip install -r requirements.txt

# Or using conda
conda install --file requirements.txt
```

### 3. AWS Configuration
Configure AWS credentials for Bedrock access:

#### Option A: AWS CLI Configuration (Recommended)
```bash
# Install AWS CLI if not already installed
pip install awscli

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and preferred region
```

#### Option B: Environment Variables
Create a `.env` file in the project root:

```bash
# Create .env file
touch .env

# Add AWS credentials (optional - use if not using AWS CLI)
echo "AWS_DEFAULT_REGION=us-east-1" >> .env
echo "AWS_PROFILE=your-aws-profile" >> .env
echo "BEDROCK_ASSUME_ROLE=arn:aws:iam::account:role/BedrockRole" >> .env
```

**Required AWS Configuration:**
- **AWS Credentials**: Access Key ID and Secret Access Key with Bedrock permissions
- **AWS Region**: Region where Bedrock is available (e.g., `us-east-1`, `us-west-2`)
- **Bedrock Permissions**: IAM permissions for `bedrock:InvokeModel` action

**AWS Setup Steps:**
1. **Create AWS Account**: Sign up at [AWS Console](https://aws.amazon.com/)
2. **Enable Bedrock**: Navigate to AWS Bedrock service and request access to Claude 3.7 Sonnet model
3. **Create IAM User**: Create user with programmatic access and Bedrock permissions
4. **Configure Credentials**: Use AWS CLI or environment variables as shown above

**Important Notes:**
- **Model Access**: Claude 3.7 Sonnet requires explicit access request in AWS Bedrock console
- **Regional Availability**: Ensure Claude 3.7 Sonnet is available in your chosen AWS region
- **Billing**: Monitor AWS costs as Bedrock charges per token usage

### 4. Verify Installation
```bash
# Test the setup (ensure AWS credentials are configured)
python main.py --generate-sample --output test_run
```

**Note**: The application will automatically create a Bedrock client and display connection information when starting.

## 🚀 Usage Guide

### 📊 Sample Dataset Generation

Generate and process a synthetic imbalanced dataset:

```bash
# Basic usage with default 10:1 ratio
python main.py --generate-sample --output results

# Custom imbalance ratio
python main.py --generate-sample --output results --imbalance-ratio 15.0

# Specific output directory
python main.py --generate-sample --output my_experiment --imbalance-ratio 5.0
```

**Sample Dataset Features:**
- 5 numeric features (feature_0 to feature_4)
- Binary target variable (0/1)
- Configurable imbalance ratios
- 1000+ samples for reliable analysis

### 📁 External Dataset Processing

Process your own CSV datasets:

```bash
# Basic usage
python main.py --file data/your_dataset.csv --target target_column --output results

# Real-world example
python main.py --file data/output.csv --target emerging_expert --output analysis_results

# With custom output directory
python main.py --file /path/to/dataset.csv --target class_label --output custom_output
```

**Supported Dataset Formats:**
- CSV files with headers
- Mixed data types (numeric + categorical)
- Missing values (automatically handled)
- Any number of features
- Binary or multi-class targets

### 📋 Command Line Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--file` | str | Path to CSV dataset | `data/dataset.csv` |
| `--target` | str | Target column name | `target`, `class`, `label` |
| `--output` | str | Output directory | `results`, `output` |
| `--generate-sample` | flag | Generate synthetic data | N/A |
| `--imbalance-ratio` | float | Imbalance ratio for synthetic data | `5.0`, `10.0`, `20.0` |

## 📈 Expected Console Output

### Sample Dataset Example
```
Create new client
  Using region: us-east-1
boto3 Bedrock client successfully created!
https://bedrock-runtime.us-east-1.amazonaws.com

Generated dataset: 1142 samples, ratio 7.0:1
Processing: results/sample_dataset.csv (target: target)
Starting data preprocessing for resampling...
Original dataset shape: (1142, 6)
Final preprocessed dataset shape: (1142, 5)
Data retention: 1142/1142 samples (100.0%)

Workflow Status: success
Imbalance: moderate (ratio: 7.04)
Applied: SMOTE
Shape: (1142, 6) → (2000, 6)
Visualizations: 2 files created
Saved: results/resampled_data.csv
ML Algorithms: 3 recommended
  1. Random Forest - Suitable for moderate datasets with good overfitting resistance
  2. XGBoost - Excellent performance on SMOTE-resampled data with efficient training
  3. Logistic Regression - Simple and interpretable for datasets with few features
Results saved to: results/result.json
```

## 📁 Output Files & Structure

The application generates comprehensive outputs in your specified directory:

```
output_directory/
├── 📊 original_distribution.png     # Class distribution before resampling
├── 📊 resampled_distribution.png    # Class distribution after resampling
├── 💾 resampled_data.csv           # Balanced dataset ready for ML training
├── 📄 result.json                  # Complete workflow results and metadata
└── 📄 sample_dataset.csv           # Generated sample data (if using --generate-sample)
```

### 📄 JSON Result Structure
```json
{
  "recommend_ml_algorithm": {
    "file_path": "results/sample_dataset.csv",
    "target_column": "target",
    "imbalance_info": {
      "imbalance_ratio": 7.04,
      "severity": "moderate",
      "is_imbalanced": true
    },
    "applied_technique": "SMOTE",
    "original_shape": [1142, 6],
    "resampled_shape": [2000, 6],
    "ml_algorithm_recommendations": {
      "algorithms": [
        {
          "name": "Random Forest",
          "reason": "Suitable for moderate datasets with good overfitting resistance"
        },
        {
          "name": "XGBoost",
          "reason": "Excellent performance on SMOTE-resampled data"
        }
      ],
      "source": "LLM"
    }
  }
}
```

## 🔧 Advanced Features

### 🤖 Intelligent Resampling Techniques

The application supports 6 advanced resampling algorithms:

| Technique | Type | Best For | Description |
|-----------|------|----------|-------------|
| **SMOTE** | Over-sampling | Moderate imbalance | Synthetic Minority Oversampling Technique |
| **Random Over-sampling** | Over-sampling | Severe imbalance | Simple duplication of minority samples |
| **Random Under-sampling** | Under-sampling | Mild imbalance | Random removal of majority samples |
| **NearMiss** | Under-sampling | Large datasets | Intelligent majority sample selection |
| **SMOTEENN** | Combined | Complex datasets | SMOTE + Edited Nearest Neighbours |
| **SMOTETomek** | Combined | Noisy datasets | SMOTE + Tomek link removal |

### 🎯 ML Algorithm Recommendations

The system provides intelligent ML algorithm suggestions based on:

**Dataset Characteristics Analyzed:**
- Dataset size (small: <10k, medium: 10k-50k, large: >50k)
- Number of features and feature types
- Applied resampling technique impact
- Computational efficiency requirements

**Supported Algorithms:**
- 🌳 **Random Forest**: Robust ensemble method, good for most datasets
- 🔍 **SVM**: Effective for high-dimensional data, computationally intensive
- 📈 **Logistic Regression**: Simple, interpretable, fast training
- 🚀 **XGBoost**: High-performance gradient boosting, excellent for tabular data
- 🧠 **Neural Networks**: Powerful but requires large datasets
- 📊 **Naive Bayes**: Fast, works well with limited data

### 🔄 Advanced Data Preprocessing

**Missing Value Handling:**
- **Numeric features**: KNN imputation (small datasets) or median imputation (large datasets)
- **Categorical features**: Mode imputation with intelligent fallbacks
- **Target variable**: Automatic row removal with data retention tracking

**Feature Engineering:**
- **Categorical encoding**: Label encoding for all categorical variables
- **Data type detection**: Automatic numeric/categorical classification
- **Feature preservation**: Maintains original column relationships

**Quality Assurance:**
- **Data retention tracking**: Reports percentage of data preserved
- **Validation checks**: Ensures data integrity throughout processing
- **Error handling**: Graceful degradation with informative messages

## 🤖 LLM Integration: AWS Bedrock + Claude 3.7 Sonnet

### Why Claude 3.7 Sonnet?
This application leverages Anthropic's **Claude 3.7 Sonnet** model through AWS Bedrock's enterprise-grade infrastructure:

**Key Advantages:**
- 🧠 **Superior Reasoning**: State-of-the-art analytical capabilities for complex data science decisions
- 🔒 **Enterprise Security**: AWS Bedrock provides enterprise-grade security and compliance
- 🌐 **Global Availability**: Deployed across multiple AWS regions for low latency
- 🎯 **Instruction Following**: Exceptional ability to follow detailed analytical instructions
- 💼 **Enterprise Ready**: Built for production workloads with SLA guarantees
- 📊 **Data Privacy**: No training on customer data, ensuring complete privacy

### LLM Configuration
```python
# Optimized settings for high-quality analytical outputs
ChatBedrock(
    model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    model_kwargs={
        "temperature": 0.1,  # Low temperature for consistent reasoning
        "max_tokens": 500    # Sufficient for detailed analysis
    },
    client=boto3_bedrock     # AWS Bedrock client with retry logic
)
```

### AWS Bedrock Client Setup
```python
# Robust client configuration with automatic retry and role assumption
boto3_bedrock = get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)
```

### Intelligent Fallback System
- **Primary**: Claude 3.7 Sonnet-powered recommendations with detailed reasoning
- **Fallback**: Rule-based recommendations when Bedrock is unavailable
- **Retry Logic**: Automatic retry with exponential backoff for transient failures
- **Reliability**: Ensures workflow completion regardless of AWS service status

## 🛠️ Troubleshooting

### Common Issues & Solutions

**❌ AWS Credentials Not Found**
```bash
NoCredentialsError: Unable to locate credentials
```
**Solution:** Configure AWS credentials using one of these methods:
```bash
# Method 1: AWS CLI
aws configure

# Method 2: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Method 3: AWS Profile
export AWS_PROFILE=your-profile-name
```

**❌ Bedrock Access Denied**
```bash
AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel
```
**Solution:** Ensure your AWS user/role has Bedrock permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-7-sonnet*"
        }
    ]
}
```

**❌ Model Not Available in Region**
```bash
ValidationException: The model ID is not supported in this region
```
**Solution:** Use a supported AWS region for Claude 3.7 Sonnet:
```bash
export AWS_DEFAULT_REGION=us-east-1  # or us-west-2, eu-west-1
```

**❌ File Not Found Error**
```bash
FileNotFoundError: File not found: data/your_file.csv
```
**Solution:** Verify file path and ensure CSV file exists:
```bash
ls -la data/  # Check if file exists
python main.py --file "$(pwd)/data/your_file.csv" --target target_col --output results
```

**❌ Target Column Not Found**
```bash
ValueError: Target column 'wrong_name' not found in dataset
```
**Solution:** Check column names in your CSV:
```python
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.columns.tolist())  # See all column names
```

**❌ Memory Issues with Large Datasets**
```bash
MemoryError: Unable to allocate array
```
**Solution:** Use data sampling or increase system memory:
```python
# Sample large datasets before processing
df_sample = df.sample(n=10000, random_state=42)
```

### Performance Optimization

**For Large Datasets (>100k samples):**
- Use `--output` on SSD storage for faster I/O
- Ensure 8GB+ RAM availability
- Consider data sampling for initial analysis
- Monitor AWS Bedrock token usage and costs

**For Many Features (>100 columns):**
- Preprocessing may take longer due to categorical encoding
- Monitor memory usage during feature transformation
- Consider feature selection before resampling
- Claude 3.7 Sonnet handles complex feature analysis efficiently

**AWS-Specific Optimizations:**
- Use AWS regions closest to your location for lower latency
- Consider AWS Bedrock provisioned throughput for high-volume usage
- Monitor CloudWatch metrics for Bedrock API performance

## 🚀 Extending the Application

### Adding New Resampling Techniques
```python
# In nodes.py, extend the techniques dictionary
techniques = {
    "SMOTE": SMOTE(random_state=42),
    "Your_New_Technique": YourNewSampler(random_state=42),
    # ... existing techniques
}
```

### Custom ML Algorithm Recommendations
```python
# In nodes.py, modify get_default_ml_algorithms()
def get_default_ml_algorithms(dataset_size: int, num_features: int) -> Dict:
    # Add your custom logic here
    if your_custom_condition:
        algorithms.append({"name": "Custom Algorithm", "reason": "Your reasoning"})
```

### Integration Ideas
- **Web Interface**: Flask/FastAPI dashboard for interactive analysis
- **Model Training**: Automatic model training and evaluation post-resampling
- **Batch Processing**: Process multiple datasets in parallel with AWS Lambda
- **Real-time Monitoring**: Track data drift and rebalancing needs with CloudWatch
- **Custom Visualizations**: Advanced plotting with Plotly/Bokeh
- **AWS Integration**: S3 for data storage, SageMaker for model deployment
- **Multi-Model Support**: Integrate other Bedrock models for specialized tasks

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/chaitanyacherukuri/Handling-Imbalanced-Data/issues)
- **Documentation**: This README and inline code comments
- **Community**: Discussions in GitHub repository

---

**Built with ❤️ using LangGraph, AWS Bedrock, Claude 3.7 Sonnet, and modern ML practices**
