# Anomaly-Detection-In-Network-Traffic
An advanced machine learning-powered web application for detecting anomalies in network traffic using models like Isolation Forest, Autoencoder, CatBoost, and more — built with Streamlit.

## 🔐 Login Credentials

To access the system, please use the following demo credentials:

- **Username**: `admin`  
- **Password**: `admin123`

## 📂 Upload Format
Upload a **cleaned CSV file** similar to the preprocessed KDD dataset.  
Use the sample below to test the system:

🔗 [Download Sample Dataset (`kddcup_clean.csv`)](https://raw.githubusercontent.com/KhushbooBoolchandani/Anomaly-Detection-In-Network-Traffic/main/kddcup_clean.csv)

> Make sure your dataset includes features like:  
`duration, protocol_type, service, flag, src_bytes, dst_bytes, ...` etc

## 🔗 Run the App
Follow the installation steps below, then open your browser and go to:  
# Run the Streamlit app
streamlit run App.py
🔗 [http://localhost:8501](http://localhost:8501)
 Network URL: http://192.168.29.104:8501
This will launch the app on your local system using Streamlit.
## 🧠 Machine Learning Models Used

### ✅ **Unsupervised Models:**
- Isolation Forest  
- Autoencoder Neural Network  

### ✅ **Supervised Models:**
- CatBoost Classifier  
- LightGBM  
- Gradient Boosting  
- Extra Trees Classifier  

Each model's performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve
- Confusion Matrix

## 📊 Features of the App

- 📁 Upload your network traffic CSV
- 🤖 Select and run multiple anomaly detection models
- 📈 Visualize model performance with charts
- 🔍 Explore predictions and anomalies detected
- 🔐 Secure login to access the app

## 🛠️ Technologies Used

- Python
- Streamlit (Frontend UI)
- Scikit-learn, TensorFlow, CatBoost, LightGBM
- Pandas, Seaborn, Matplotlib
- Joblib (Model Persistence)
- Streamlit Option Menu for sidebar navigation

## 📦 Installation Instructions

```bash
# Clone the repository
git clone https://github.com/KhushbooBoolchandani/Anomaly-Detection-In-Network-Traffic.git
cd Anomaly-Detection-In-Network-Traffic

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run App.py
