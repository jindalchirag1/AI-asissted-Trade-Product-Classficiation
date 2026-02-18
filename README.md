# Automated Classification of Tax Products HS Code using Transformers and Gradient Boosting

## Project Overview

This repository contains the architectural framework and methodology for my thesis project: **Automated Commodity Classification**. The goal of this research is to classify Chilean trade descriptions into the 6-digit Harmonized System (HS6) codes using state-of-the-art Natural Language Processing (NLP).

**Note:** The dataset used in this research is proprietary/private and is not included in this repository. The source code provided serves as a structural reference for the methodologies discussed in the thesis.

##  Key Research Questions

1. How do Spanish-specific Transformer models (BETO) compare to general multilingual models in trade classification?  
2. Can Gradient Boosting (XGBoost/LightGBM) with TF-IDF features match Deep Learning performance in high-cardinality classification?  
3. What are the most effective strategies for handling extreme class imbalance in customs data?

##  Technical Architecture

### 1\. NLP Pipeline

* **Preprocessing:** Custom filtering for high-cardinality classes and noise reduction in Spanish trade descriptions.  
* **Traditional Approach:** TF-IDF Vectorization with over 5,000 features.  
* **Deep Learning Approach:** Subword tokenization using the BETO (Spanish BERT) and RoBERTa architectures.

### 2\. Models Benchmarked

* **BETO (Spanish BERT):** Pre-trained on a massive Spanish corpus, utilized for its deep understanding of regional linguistic nuances.  
* **XGBoost & LightGBM:** Implemented with **Cost-Sensitive Learning (Class Weights)** to address the "Long Tail" distribution of HS6 codes.  
* **Multilingual BERT:** Evaluated for its cross-lingual transfer capabilities.

## Evaluation Strategy

Because the dataset is highly imbalanced, success was measured using:

* **Macro F1-Score:** Our primary metric to ensure performance is consistent across rare and common products.  
* **Stratified K-Fold Validation:** To maintain representative class distributions across all folds.  
* **Precision/Recall Analysis:** To minimize misclassification in sensitive customs categories.

##  Repository Structure


```text
├── models/
│   ├── model_configs/          # Hyperparameter configurations
│   └── architecture_notes.md   # Detailed logic for model selection
├── src/
│   ├── preprocessing_logic.py  # Methodology for data cleaning (Code snippets only)
│   └── evaluation_metrics.py   # Custom metric implementations
├── README.md
└── results_summary.txt         # High-level performance comparison (No raw data)


```
##  Key Findings

* **Context over Frequency:** The BETO model significantly outperformed traditional Boosting models on descriptions containing ambiguous technical jargon.  
* **Imbalance Handling:** Class weighting proved more effective than synthetic oversampling (SMOTE) due to the presence of classes with extremely low support.  
* **Production Trade-offs:** While Transformers provided higher accuracy, Gradient Boosting models offered 10x faster inference speeds, suggesting a hybrid deployment approach.

## ** Contact & Inquiries**

This project was developed as part of my **M.Sc. Data Analytics Thesis**. If you are an employer or researcher interested in the full methodology or the specific performance metrics, please feel free to reach out.

**Skills Demonstrated:** Python • PyTorch • Hugging Face • XGBoost • LightGBM • Scikit-Learn • Spanish NLP
