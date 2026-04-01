# 🚗 SmartCar AI: Advanced Car Price Predictor

A state-of-the-art Car Price Prediction system built with **Flask**, **Random Forest (Scikit-Learn)**, and a modern **Glassmorphic UI**. This project utilizes high-quality data (2015-2025) and dynamic machine learning pipelines to provide accurate valuations.

---

## 🚀 Key Features

*   **Premium Glassmorphic UI**: Transparent, blurred interfaces with high contrast and animations.
*   **Dynamic Model Filtering**: Brand-specific models populate automatically based on user choice.
*   **Real-time AI Prediction**: Uses a **Random Forest Regressor** to predict prices with high R² accuracy.
*   **Dynamic Dataset Upload**: Upload a clean CSV to automatically retrain the ML model.
*   **Interactive Insights**: View market trends using Chart.js visualizations.
*   **Full Responsive Layout**: Optimized for Desktop and Mobile.


---

## 📂 Project Structure

```text
├── app.py                # Flask Backend API
├── templates/
│   └── index.html        # Modern Glassmorphic Frontend
├── static/
│   └── css/              # Custom Styles (if any)
├── dataset.csv           # Modern Car Dataset (6000+ entries)
├── model.pkl             # Trained Pickle Model
├── requirements.txt      # Project Dependencies
└── README.md             # Documentation
```

---

## 🛠️ Tech Stack

*   **Frontend**: HTML5, CSS3, Bootstrap 5, Chart.js, Animate.css, FontAwesome.
*   **Backend**: Flask (Python).
*   **Machine Learning**: Scikit-Learn, Pandas, NumPy, Pickle.
*   **Algorithm**: Random Forest Regressor (Chosen for its robustness and accuracy with multi-category features).

---

## ⚙️ Installation & Running Locally

1.  **Clone the Repository** (or download the files).
2.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Application**:
    ```bash
    python app.py
    ```
4.  **Navigate to Host**: Open `http://127.0.0.1:5000/` in your browser.

---

## 🧠 Model Pipeline

1.  **Data Loading**: Cleans and validates input CSV.
2.  **Preprocessing**: Uses `OneHotEncoder` via `ColumnTransformer` to handle categorical data (Brand, Model, Fuel, etc.).
3.  **Training**: Implements a **Random Forest Regressor** with 100 estimators.
4.  **Serialization**: Saves the pipeline as `model.pkl` for fast inference.

---

## 📊 Dataset Specifications (2015 - 2025)

The included dataset is meticulously generated and contains features like:
*   `company`, `model`, `year`, `fuel_type`, `transmission`, `km_driven`, `owner_type`, `mileage`, `engine`, and `price`.

---

## ✨ Developed by Antigravity (Advanced AI Coding Assistant)

*   **Status**: Production Ready & LinkedIn-worthy.
*   **Category**: Machine Learning / Full-Stack Web Development.
