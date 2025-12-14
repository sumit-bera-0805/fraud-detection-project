# 💳 Fraud Detection Project (Machine Learning)

An **end-to-End Fraud Detection System** built using Machine Learning to identify suspicious credit card transactions.  
This project demonstrates a **complete data analyst / ML workflow** from raw data to model training and an interactive dashboard.

---

## 🚀 Project Overview

Credit card fraud is a major real-world problem.  
In this project, we:

- Analyze transaction data
- Handle extreme class imbalance
- Train a high-performance ML model
- Deploy results using an interactive Streamlit dashboard

The system predicts **fraud probability** for each transaction and highlights the most suspicious ones.

---

## 🧠 Key Features

- Data preprocessing & feature engineering  
- Class imbalance handling using **SMOTE**  
- **LightGBM** model for high accuracy  
- Model evaluation with AUC metric  
- Interactive **Streamlit dashboard**  
- Clean, modular project structure  

---

## 📂 Project Structure

fraud-detection/
│
├── app/
│ └── streamlit_app.py # Streamlit dashboard
│
├── src/
│ ├── data_loader.py # Data loading & split
│ ├── features.py # Feature engineering
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation
│ └── serve.py # API (optional)
│
├── data/
│ └── creditcard.csv # Dataset
│
├── models/
│ └── fraud_model.pkl # Trained model
│
├── screenshots/ # Dashboard screenshots
│
├── README.md
└── requirements.txt

yaml
Copy code

---

## 🧪 Machine Learning Pipeline

1. Load credit card transaction dataset  
2. Feature engineering (scaling, log transform)  
3. Handle class imbalance using **SMOTE**  
4. Train **LightGBM** model  
5. Save trained model  
6. Predict fraud probability  
7. Visualize results using Streamlit  

---

## 📊 Model Details

- **Algorithm:** LightGBM (Gradient Boosting)
- **Problem Type:** Binary Classification
- **Evaluation Metric:** AUC (Area Under Curve)
- **Why LightGBM?**
  - Fast training
  - Handles large datasets
  - Performs well on imbalanced data

---

## 🖥️ How to Run the Project (Step-by-Step)

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Train the Model
bash
Copy code
python src/train.py
You should see:

nginx
Copy code
Model training complete!
4️⃣ Run Streamlit Dashboard
bash
Copy code
streamlit run app/streamlit_app.py
Open browser:

arduino
Copy code
http://localhost:8501
📸 Screenshots (Dashboard Preview)
Upload & Prediction View

Fraud Probability Output

📌 How to add screenshots

Take screenshot of Streamlit dashboard

Create a folder named screenshots in project root

Save images as:

dashboard_main.png

fraud_results.png

Commit & push to GitHub

📌 Output
Fraud probability score for each transaction

Top suspicious transactions highlighted

Interactive & easy-to-use dashboard

📄 Resume Project Description (Use This)
Fraud Detection System using Machine Learning
Built an end-to-end fraud detection pipeline using Python, SMOTE, and LightGBM.
Performed feature engineering, handled class imbalance, trained a high-accuracy model, and deployed predictions through an interactive Streamlit dashboard.

👨‍💻 Author
Sumit Bera
GitHub: https://github.com/sumit-bera-0805

⭐ If you like this project, give it a star on GitHub!
