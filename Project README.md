# **Automated Classification of Tax Products HS Code using Transformers and Gradient Boosting**

## **📌 Project Overview**

This repository contains the architectural framework and methodology for my thesis project: **Automated Commodity Classification**. The goal of this research is to classify Chilean trade descriptions into the 6-digit Harmonized System (HS6) codes using state-of-the-art Natural Language Processing (NLP).

**Note:** The dataset used in this research is proprietary/private and is not included in this repository. The source code provided serves as a structural reference for the methodologies discussed in the thesis.

## **🚀 Key Research Questions**

1. How do Spanish-specific Transformer models (BETO) compare to general multilingual models in trade classification?  
2. Can Gradient Boosting (XGBoost/LightGBM) with TF-IDF features match Deep Learning performance in high-cardinality classification?  
3. What are the most effective strategies for handling extreme class imbalance in customs data?

## **🏗️ Technical Architecture**

### **1\. NLP Pipeline**

* **Preprocessing:** Custom filtering for high-cardinality classes and noise reduction in Spanish trade descriptions.  
* **Traditional Approach:** TF-IDF Vectorization (![][image1]\-grams 1-2) with over 5,000 features.  
* **Deep Learning Approach:** Subword tokenization using the BETO (Spanish BERT) and RoBERTa architectures.

### **2\. Models Benchmarked**

* **BETO (Spanish BERT):** Pre-trained on a massive Spanish corpus, utilized for its deep understanding of regional linguistic nuances.  
* **XGBoost & LightGBM:** Implemented with **Cost-Sensitive Learning (Class Weights)** to address the "Long Tail" distribution of HS6 codes.  
* **Multilingual BERT:** Evaluated for its cross-lingual transfer capabilities.

## **📊 Evaluation Strategy**

Because the dataset is highly imbalanced, success was measured using:

* **Macro F1-Score:** Our primary metric to ensure performance is consistent across rare and common products.  
* **Stratified K-Fold Validation:** To maintain representative class distributions across all folds.  
* **Precision/Recall Analysis:** To minimize misclassification in sensitive customs categories.

## **🛠️ Repository Structure**

### 🛠️ Repository Structure
```text
├── 📂 models/
│   ├── 📂 model_configs/      # Hyperparameter configurations
│   └── 📝 architecture_notes.md  # Detailed logic for model selection
├── 📂 src/
│   ├── 🐍 preprocessing_logic.py # Methodology for data cleaning
│   └── 🐍 evaluation_metrics.py   # Custom metric implementations
├── 📝 README.md
└── 📊 results_summary.txt        # High-level performance comparison
## **💡 Key Findings**

* **Context over Frequency:** The BETO model significantly outperformed traditional Boosting models on descriptions containing ambiguous technical jargon.  
* **Imbalance Handling:** Class weighting proved more effective than synthetic oversampling (SMOTE) due to the presence of classes with extremely low support (![][image2]).  
* **Production Trade-offs:** While Transformers provided higher accuracy, Gradient Boosting models offered 10x faster inference speeds, suggesting a hybrid deployment approach.

## **👤 Contact & Inquiries**

This project was developed as part of my **AI Engineering Thesis**. If you are an employer or researcher interested in the full methodology or the specific performance metrics, please feel free to reach out.

**Skills Demonstrated:** Python • PyTorch • Hugging Face • XGBoost • LightGBM • Scikit-Learn • Spanish NLP

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAA2UlEQVR4XmNgGAWDCigrK4sBKWYYX0FBQQBdDAyUlJT45eXlt8jJyfkC6f9AfAaIl4HkpKWlhYHsc0B8Fa4BqNAFKLAcaGI6VMMekCEgOVFRUR4g/wAQP4RrAHI8oXgNEP8DavRAMkwJKPYcxQYYAAr+BuKtQA0cUCEWkM3ohsAByDlAE8uR+IpA/ASIrysqKooD5WJUVFTYYfIg034DBW1gAkh+aoUaMBlIMYIlgQqNgXg+XAAIgKbqAxW9BuL9QHwRJg4DzMbGxqzogiD/ABVLIvlrFNAGAAAdLzNIUwIiSwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAZCAYAAACRiGY9AAACuUlEQVR4Xu2WO2iTURTHE6Ig+MJHCM3r5iVBcRCCgqWjW6hDcRAUBAVdfGwVVNTFQQSHoEspiAVRHHQxFbRgsQ4RXJzq7FCoULrooIL1d/LdLzm55NGmDYJ8f/hzv++c/73fOffcxxcKBQgQYGDIZDKHjDEz6XT6K+0X2kuYw1qD/Tq6x7QTPqWf1gwa2Ww2RhNxzPLeEmsdyWQyQSIn4BWCXYHLqVTqoJKEsQ3jv0z7g/YmbTkWi21VmoEiGo1u45uz8Jea2Bn4ncnd4uobQFBBcNsmNm9npoF4PL4X+35t2wgwgYdLpdJm167hJ6VWy7lEIrHH1bWgWCxuR/hKKkS7bBO7qDVUaATu0rb1QIIiyLtS+dUmhXbU9XWEVABO87iJdtImVdMaArig3/sFK6DI2FNwQWbc9bdDv0mdosMj+3wU/oR/lESSfa7e+0GEMYbhJ1iWd1fQCX5SrKTjtBU4QbynO+5rWVKIpo3aLzyPSVKSoNUco1Jvmr1WB/koYyzJhNFmXf8aIIfVFOOM+wYOuH3YvrGMk1pYB8ISzrd6v+RyuZ3YavABr2F8V3m+o7p1BdoheB8u9tzQfUKdiC17vw7ZKzgqrl3EcMEeHi+lWq6mE9BfM97xf8v1bSCkek+E8tzOMaaNAmxZuEJgz2jn5D5zNd1QKBR20O8GfAiHXP9aQAznGWNJxtN2e8TPStUaRnv3vDMd7h9JyrLa9ZLrAvqeNd5J94LgDrj+XlDLTOL47NvtfpULeFLrIwT6HmONKhRCbX438D2FvwlmxPX1CVkZr+E83z7Z637ywTVwhD4fbJwC2efjkmhjDNM8tiV7n9XmMB5ER+ePUlHXtx7k8/mU8ZbloixT198OaMuiN94fxZzxTud7ru6fw+67KhXY7fraQU5lKnyGiR51f+MCBAgQIMB/g78FFrLgDfwvwQAAAABJRU5ErkJggg==>
