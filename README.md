# 💳 AI-Powered Credit Card Fraud Detection System

An intelligent **Machine Learning-based Credit Card Fraud Detection System** built with **Python**, **Streamlit**, and **XGBoost**. The application predicts whether a transaction is fraudulent by analyzing transaction behavior and provides both **single transaction prediction** and **batch fraud detection** using CSV or Excel files. The project includes an interactive dashboard with fraud probability visualization, automated recommendations, and downloadable prediction reports.

---

## 🚀 Features

* 🔍 Real-time fraud prediction for individual transactions
* 📂 Batch fraud detection using CSV and Excel files
* 📊 Interactive fraud probability gauge
* 📈 Fraud probability distribution visualization
* 📥 Download prediction results as an Excel report
* ⚡ Automatic handling of missing values
* 🎯 Fraud risk classification with actionable recommendations
* 📱 Simple and responsive Streamlit interface

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **XGBoost**
* **Scikit-learn**
* **NumPy**
* **Pandas**
* **Plotly**
* **OpenPyXL**
* **XlsxWriter**

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py                        # Streamlit Web Application
├── fraud_detection_model.pkl     # Trained ML Model
├── use_fraud_model.py            # Model Usage Examples
├── requirements.txt              # Required Libraries
├── README.md
```

---

## 📊 Dataset Features

The model uses the following transaction attributes:

| Feature                        | Description                                         |
| ------------------------------ | --------------------------------------------------- |
| distance_from_home             | Distance from the cardholder's home location        |
| distance_from_last_transaction | Distance from the previous transaction              |
| ratio_to_median_purchase_price | Current purchase compared to user's median spending |
| repeat_retailer                | Whether the retailer has been visited before        |
| used_chip                      | Whether the card chip was used                      |
| used_pin_number                | Whether a PIN was entered                           |
| online_order                   | Whether the transaction was made online             |

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Cleaning & Preprocessing
3. Feature Selection
4. Model Training using XGBoost
5. Fraud Probability Prediction
6. Risk Classification
7. Visualization & Reporting

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Credit-Card-Fraud-Detection.git

cd Credit-Card-Fraud-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires the following packages: Streamlit, NumPy, Scikit-learn, XGBoost, Pandas, Matplotlib, Seaborn, tqdm, Plotly, XlsxWriter, and OpenPyXL.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📋 Using the Application

### Single Transaction Prediction

* Enter the transaction details.
* Click through the provided inputs.
* View:

  * Fraud Probability
  * Risk Level
  * Recommendation

### Batch Prediction

Upload a CSV or Excel file containing the required columns.

Example:

| distance_from_home | distance_from_last_transaction | ratio_to_median_purchase_price | repeat_retailer | used_chip | used_pin_number | online_order |
| ------------------ | ------------------------------ | ------------------------------ | --------------- | --------- | --------------- | ------------ |
| 37.31              | 7.12                           | 2.96                           | 1               | 1         | 0               | 1            |

After processing, the application displays:

* Total Transactions
* Fraudulent Transactions
* Average Fraud Probability
* High-Risk Transactions
* Downloadable Excel Report

---

## 📈 Risk Classification

| Fraud Probability | Recommendation                                           |
| ----------------- | -------------------------------------------------------- |
| Below 40%         | Process transaction normally                             |
| 40% – 70%         | Flag for manual review                                   |
| Above 70%         | Block transaction immediately and contact the cardholder |

---

## 📊 Dashboard Highlights

* Fraud Probability Gauge
* Transaction Statistics
* Fraud Distribution Histogram
* Interactive Tables
* Excel Report Export

---

## 📦 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

* Deep Learning Models
* Real-time API Integration
* Live Banking Transaction Monitoring
* User Authentication
* Fraud Explanation using Explainable AI (SHAP/LIME)
* Cloud Deployment (AWS / Azure / GCP)
* Model Retraining Pipeline

---

## 📚 Learning Outcomes

This project demonstrates:

* Machine Learning Classification
* Data Preprocessing
* Fraud Detection
* Feature Engineering
* Model Deployment
* Streamlit Dashboard Development
* Data Visualization
* Batch Prediction Systems

---

## 👨‍💻 Author

**Raj Srivastava**

B.Tech (Computer Science & Engineering - Data Science)

Passionate about Artificial Intelligence, Machine Learning, Data Science, and Full-Stack Development.

---

## ⭐ If you found this project helpful, don't forget to Star the repository!
