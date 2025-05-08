# 🎵 Music Genre Classifier Using PySpark and Flask

This project classifies song lyrics into music genres using a machine learning pipeline built with **PySpark (MLlib)** and an interactive **Flask web interface**. Initially inspired by [Taras Matyashovskyy's work](https://github.com/tarasmatyashovskyy), this enhanced version expands genre classification from 2 classes ("Pop", "Metal") to **8 distinct genres**.

---

## 📂 Supported Genres

* Pop
* Country
* Blues
* Jazz
* Reggae
* Rock
* Hip-Hop
* Soul

---

## 📦 Dataset

### 1. Mendeley Dataset

A publicly available dataset containing \~28,000 labeled song lyrics across 7 genres.

### 2. Student Dataset

A manually or automatically collected dataset of **at least 100 songs** from a genre not present in the Mendeley dataset (e.g., Soul).

These datasets are merged and preprocessed into a final dataset for training.

---

## ⚙️ Tech Stack

* **PySpark 3.5.5**
* **Python 3.9.0**
* **Java 11**
* **Flask**
* **Chart.js** (for result visualization)

---

## 🚀 Features

* Classifies lyrics into **8 genres** using a logistic regression model
* Interactive **Flask web app** to paste lyrics and get predictions
* **Bar chart visualization** of prediction probabilities using Chart.js
* Model training, evaluation, and saving using **Spark ML Pipelines**
* Generates and exports a **confusion matrix** (CSV + image)
* One-click setup via `run.bat`

---

## 🛠️ Installation & Setup

### 📌 Requirements

* Java 8 / 11 / 17
* Python 3.9+
* Git

### 🔧 Steps to Run

```bash
git clone https://github.com/Kajaani-Balabavan/music-genre-classifier.git
cd music-genre-classifier
run.bat
```

This will:

* Create a virtual environment (`.venv`)
* Install dependencies from `requirements.txt`
* Train the model and save it under `/models`
* Launch the Flask app at [http://localhost:5000](http://localhost:5000)

---

## 📁 Directory Structure

```
.
├── data/                        # Exported datasets and outputs
│   ├── merged_dataset.csv
│   ├── predicted_results.csv
│   ├── confusion_matrix.csv
│   └── confusion_matrix_plot.png
├── models/                      # Trained PySpark MLlib pipeline and label indexer
│   ├── label_indexer_model/
│   └── trained_model_pipeline/
├── src/                         # Source code
│   ├── main.py                  # Model training script
│   ├── app.py                   # Flask app
│   ├── templates/               # HTML templates
│   │   ├── index.html
│   │   ├── result.html
│   │   └── error.html
│   └── static/                  # Static assets
│       ├── styles.css
│       └── background.avif
├── Student_dataset.csv          # Custom genre dataset
├── mendeley_dataset.csv         # Original dataset
├── Merged_dataset.txt           # Merge logic reference (optional)
├── requirements.txt             # Python dependencies
├── run.bat                      # Batch script to run the app
├── demo.mp4                     # (Optional) Demo video
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📈 Model Evaluation

After training:

* The model is evaluated on a 20% test set
* Generates predictions alongside true labels
* Outputs a **confusion matrix** (CSV + image) for analysis

---

## 🌐 Web Interface Usage

1. Go to [http://localhost:5000](http://localhost:5000)
2. Paste lyrics into the input field
3. Click **Predict Genre**
4. View the predicted genre with a visual confidence breakdown
