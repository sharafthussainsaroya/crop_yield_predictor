# 🌾 Crop Yield Predictor — Pakistan Agriculture

A Machine Learning project to predict crop yield (Wheat, Maize, Canola) based on agronomic and environmental factors relevant to Pakistani farming conditions.

---

## 👨‍🎓 Author
**Sharaft Hussain**  
M.Sc. (Hons.) Agronomy — MNS University of Agriculture Multan  
Certified in Advance Python Programming & Applications (NAVTTC | Grade A+)

---

## 🎯 Project Objective

To build a predictive model that estimates crop yield (ton/ha) using key field parameters, helping farmers and agronomists make data-driven decisions.

---

## 🔬 Features Used for Prediction

| Feature | Description |
|---|---|
| Crop Type | Wheat / Maize / Canola |
| Rainfall (mm) | Seasonal rainfall amount |
| Temperature (°C) | Average growing temperature |
| Nitrogen (kg/ha) | Nitrogen fertilizer applied |
| Phosphorus (kg/ha) | Phosphorus fertilizer applied |
| Irrigation Times | Number of irrigations applied |
| Soil pH | Soil acidity/alkalinity |
| Sowing Density | Seeds per hectare |

---

## 🤖 Models Used

- **Linear Regression** — Baseline model
- **Random Forest Regressor** — Best performing model

---

## 📊 Results

| Model | R² Score | MAE (ton/ha) |
|---|---|---|
| Linear Regression | ~0.93 | ~0.18 |
| Random Forest | ~0.97 | ~0.12 |

---

## 📁 Project Files

```
crop_yield_predictor/
│
├── crop_yield_predictor.py   # Main Python script
├── eda_analysis.png          # EDA charts (auto-generated)
├── feature_importance.png    # Feature importance chart (auto-generated)
└── README.md                 # Project documentation
```

---

## ▶️ How to Run

```bash
# 1. Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# 2. Run the script
python crop_yield_predictor.py
```

---

## 📦 Libraries Used

```python
pandas       # Data manipulation
numpy        # Numerical computing
matplotlib   # Data visualization
seaborn      # Statistical plots
scikit-learn # Machine learning models
```

---

## 🌱 Why This Project?

As an agronomist with hands-on field research experience in intercropping systems and nitrogen management, I combined my domain knowledge with Python programming skills to build this practical ML tool that bridges the gap between agriculture science and data technology.

---

## 📬 Contact

- **Email:** sharafthussain076@gmail.com  
- **LinkedIn:** [linkedin.com/in/sharaft-hussain-saroya](https://www.linkedin.com/in/sharaft-hussain-saroya)
