# 🚀 FidelFolio Investment Challenge

## 📈 Project Overview
This repository contains the code and documentation for the FidelFolio Investment Challenge. Our goal is to predict market capitalization growth using deep learning techniques. The project includes data preprocessing, company sector classification, feature imputation, and model training for three distinct target variables.

## 📁 Files and Structure
* `Report.pdf`: Project report detailing the methodology, preprocessing pipeline, model architecture, and results
* `b.py`: Python script for classifying companies into sectors using Ollama LLM
* `preprocessing.ipynb`: Jupyter notebook for data cleaning, feature imputation, and preprocessing
* `model.ipynb`: Jupyter notebook containing model architectures, training procedures, and evaluation

## ⚙️ Prerequisites
To run this project, you need the following:
* Python 3.8+
* PyTorch with CUDA support (for GPU acceleration)
* Pandas, NumPy, scikit-learn, tqdm, matplotlib, seaborn
* sentence-transformers library
* Ollama LLM (for sector classification)
* Jupyter Notebook or Google Colab (for running notebooks)

Install required Python packages:
```bash
pip install pandas numpy torch scikit-learn tqdm matplotlib seaborn sentence-transformers requests
```

## 🛠️ Setup Instructions

### 1. Data Preparation
Ensure you have the input data files:
* `company_sec.csv`: Main dataset containing company features and targets
* `mid.csv` or `companies.txt`: Company list for sector classification

### 2. Ollama LLM Setup
* Install Ollama following the official instructions: https://ollama.ai/
* Run the Ollama server locally:
```bash
ollama run llama
```

### 3. Environment Configuration
* Create a virtual environment (recommended):
```bash
python -m venv fidelfolio-env
source fidelfolio-env/bin/activate  # On Windows: fidelfolio-env\Scripts\activate
```
* Install the required packages

## 🏃‍♂️ Running the Code

### Step 1: Sector Classification
Run the `b.py` script to classify companies into sectors:
```bash
python b.py
```

This will:
* Load company names from `mid.csv` or `companies.txt`
* Classify companies into predefined sectors using Ollama LLM
* Save results to `company_sectors_llama3.csv`

### Step 2: Data Preprocessing
Open and run the `preprocessing.ipynb` notebook:
1. Load and clean the data from `company_sec.csv`
2. Compute sector embeddings using SentenceTransformer
3. Impute missing values using hybrid similarity approach
4. Generate imputation logs and summary plots

### Step 3: Model Training
Open and run the `model.ipynb` notebook:
1. Prepare data for each target variable
2. Define model architectures (HybridTemporalNet, BalancedModel)
3. Train models for Target 1, Target 2, and Target 3
4. Evaluate models and save trained weights

## 📊 Results and Outputs
* Preprocessing outputs:
   * `imputation_log_*.csv`: Log of imputed values and methods used
   * `hybrid_imputation_details_*.csv`: Detailed information about hybrid imputation
   * `imputation_summary_*.png`: Summary plot of imputation methods distribution
* Model training outputs:
   * Trained model weights for each target
   * Training history and evaluation metrics
   * Average RMSE values across companies

## 📝 Notes and Considerations
* The code assumes temporal ordering of data and maintains chronological integrity
* For large datasets, consider optimizing batch sizes and using more powerful GPU
* The sector classification step requires a local Ollama instance
* Model performance can be improved by hyperparameter tuning and longer training
* The hybrid imputation strategy balances sector similarity and feature proximity

## 🤝 Contributing
We welcome contributions! Please feel free to submit a Pull Request.

## ❓ Support
For questions or issues, please refer to the project report or contact the authors.

## 🎉 Acknowledgements
Special thanks to all team members who contributed to this challenge!

---

Happy coding! 💻🚀
