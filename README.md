# 📰 News Article Classifier (NLP + Flask)

An end-to-end Natural Language Processing (NLP) project that classifies news articles into categories like **Technology, Politics, Business, Sports, and Entertainment** using Machine Learning. The project also includes a **Flask web application** for real-time predictions.

---

## 🚀 Features

* Classifies news articles into multiple categories
* Uses **TF-IDF** for text feature extraction
* Trained using **Multinomial Naive Bayes**
* Real-time prediction through a **Flask web app**
* Input validation for better user experience
* Confidence-based filtering for reliable predictions

---

## 🛠️ Tech Stack

* **Python**
* **Scikit-learn**
* **NLTK**
* **Pandas**
* **Matplotlib**
* **Flask**

---

## 📂 Project Structure

```
News-Article-Classifier/
│── app.py
│── model.pkl
│── vectorizer.pkl
│── bbc-text.csv
│── train_model.py
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## ⚙️ How It Works

1. Text input is cleaned (lowercase, remove punctuation, stopwords, etc.)
2. Converted into numerical format using **TF-IDF vectorization**
3. Passed to the trained **Naive Bayes model**
4. Model predicts category with confidence score
5. Flask app displays result to the user

---

## ▶️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/news-article-classifier.git
cd news-article-classifier
```

### 2. Download NLTK stopwords

```python
import nltk
nltk.download('stopwords')
```
### 3. Run the model 

```bash
python train_model.py
```

### 4. Run the application

```bash
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000
```

---

## 📊 Model Performance

* Algorithm: **Multinomial Naive Bayes**
* Feature Extraction: **TF-IDF**
* Accuracy: **96%**
---

## 🧪 Example Input

> "The government has announced new policies to improve the economy..."

👉 Output: **Politics**

---

## 📌 Future Improvements

* Deploy the app online (Render / Railway)
* Try advanced models (Logistic Regression, SVM, Deep Learning)
* Add confidence score display in UI
* Improve UI/UX design

---

## 👩‍💻 Author

**Jyoti Patel**
