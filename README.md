# 📰 News Classification Project

This project classifies news into categories based on their headlines. Several machine learning models were tested:

- Decision Tree
- Support Vector Classifier (SVC)
- Multinomial Naive Bayes (best performer)
- Multilayer Perceptron (MLP)
- Random Forest

🔍 **Why Naive Bayes?**  
Multinomial Naive Bayes performed the best — which makes sense, as it effectively classifies based on keywords, much like how humans do.

---

## ⚙️ Installation & Setup

This project is designed to run on **Google Colab** — no local installation needed.

### Steps:

1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Upload the dataset (if required)
3. Run the cells to train and evaluate different classifiers

---

## 📁 Files

- `news_classification.ipynb` – main notebook with code and results
- `news_dataset.csv` – dataset containing news headlines and labels (optional upload)

---

## ✅ Results

- **Best Model:** Multinomial Naive Bayes
- **Reason:** Performs well for text classification based on word frequency and independence assumptions.
