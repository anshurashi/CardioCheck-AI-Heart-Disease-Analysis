# ❤️ CardioCheck: KNN-Powered Heart Risk Analysis

This AI tool utilizes a **K-Nearest Neighbors (KNN)** algorithm to predict the likelihood of heart disease based on 11 clinical features.

## 📊 Model Performance
- **Algorithm:** KNN (K-Nearest Neighbors)
- **Scaling:** RobustScaler / StandardScaler
- **Key Features:** ST Slope, Chest Pain Type, Max Heart Rate.

## 🚀 How to Run
1. Ensure `knn_heart.pkl`, `scaler.pkl`, and `columns.pkl` are in the root directory.
2. Install dependencies: `pip install streamlit pandas joblib scikit-learn`
3. Launch: `streamlit run app.py`

## 🧪 Clinical Parameters
- **ST Slope:** The slope of the peak exercise ST segment.
- **Oldpeak:** ST depression induced by exercise relative to rest.
- **Chest Pain Type:** TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic.
