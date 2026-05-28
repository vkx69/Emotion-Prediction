# 😊 Emotion Detection App

A Machine Learning + NLP based web application that predicts human emotions from text using Streamlit.

## 🚀 Overview

This project detects emotions from user input text such as:

* Joy 😊
* Sadness 😔
* Anger 😡
* Love ❤️
* Fear 😨
* Surprise 😲

The application is built using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.

---

## 🛠 Tech Stack

* Python
* Streamlit
* Scikit-learn
* NLP (TF-IDF Vectorization)
* Logistic Regression
* Pandas
* Joblib

---

## ⚙️ How it Works

1. User enters text
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical vectors
4. Logistic Regression predicts the emotion
5. Result is displayed in the Streamlit UI

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Example

Input:

```text
I am feeling very happy today
```

Output:

```text
Joy 😊
```

---

## ⚠️ Note

This project is based on **NLP + Machine Learning**, so predictions depend on the training dataset and context of the sentence.

Sometimes short or unclear sentences may be predicted differently.

Example:

```text
Input: I am not good
Predicted: Surprise 😲
Expected: Sadness 😔
```

This happens because ML models learn patterns from training data and may misclassify text when the context is limited or ambiguous.

Improving the dataset or using deep learning models can improve accuracy.

---

## Future Improvements

* Better dataset training
* Higher prediction accuracy
* Deep Learning / LSTM / BERT implementation
* Deployment on Streamlit Cloud

---
👨‍💻 Author
Vikas Kumar
