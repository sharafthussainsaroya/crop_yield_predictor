# ============================================================
# CROP YIELD PREDICTOR - Pakistan Agriculture Data
# Author  : Sharaft Hussain
# Tools   : Python, Pandas, Scikit-learn, Matplotlib, Seaborn
# Purpose : Predict wheat/maize/canola yield using ML model
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET (Pakistan crops)
# ──────────────────────────────────────────────
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'Crop':             np.random.choice(['Wheat', 'Maize', 'Canola'], n),
    'Rainfall_mm':      np.random.uniform(150, 600, n),
    'Temperature_C':    np.random.uniform(18, 38, n),
    'Nitrogen_kg_ha':   np.random.uniform(60, 180, n),
    'Phosphorus_kg_ha': np.random.uniform(30, 90, n),
    'Irrigation_times': np.random.randint(2, 8, n),
    'Soil_pH':          np.random.uniform(6.5, 8.5, n),
    'Sowing_density':   np.random.uniform(80, 150, n),
})

# Yield formula (realistic agronomic logic)
def calculate_yield(row):
    base = {'Wheat': 3.5, 'Maize': 5.0, 'Canola': 2.0}[row['Crop']]
    y = (base
         + 0.003  * row['Rainfall_mm']
         + 0.01   * row['Nitrogen_kg_ha']
         + 0.005  * row['Phosphorus_kg_ha']
         + 0.15   * row['Irrigation_times']
         - 0.04   * abs(row['Temperature_C'] - 25)
         - 0.1    * abs(row['Soil_pH'] - 7.2)
         + np.random.normal(0, 0.3))
    return round(max(y, 0.5), 2)

data['Yield_ton_ha'] = data.apply(calculate_yield, axis=1)

print("=" * 55)
print("   CROP YIELD PREDICTOR — Pakistan Agriculture Data")
print("=" * 55)
print(f"\n Dataset size : {len(data)} records")
print(f" Crops        : {data['Crop'].unique()}")
print(f"\n{data.describe().round(2)}")

# ──────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Crop Yield Analysis — Pakistan Agriculture\nBy: Sharaft Hussain',
             fontsize=14, fontweight='bold', color='#1A6B5F')

# Plot 1 — Average yield by crop
avg_yield = data.groupby('Crop')['Yield_ton_ha'].mean().sort_values()
colors = ['#1A6B5F', '#2E9E8F', '#A8D5CF']
axes[0, 0].barh(avg_yield.index, avg_yield.values, color=colors)
axes[0, 0].set_title('Average Yield by Crop (ton/ha)', fontweight='bold')
axes[0, 0].set_xlabel('Yield (ton/ha)')
for i, v in enumerate(avg_yield.values):
    axes[0, 0].text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=10)

# Plot 2 — Nitrogen vs Yield scatter
for crop, color in zip(['Wheat', 'Maize', 'Canola'], ['#1A6B5F', '#E07B39', '#8B5E3C']):
    subset = data[data['Crop'] == crop]
    axes[0, 1].scatter(subset['Nitrogen_kg_ha'], subset['Yield_ton_ha'],
                       label=crop, alpha=0.6, color=color, s=40)
axes[0, 1].set_title('Nitrogen vs Yield', fontweight='bold')
axes[0, 1].set_xlabel('Nitrogen (kg/ha)')
axes[0, 1].set_ylabel('Yield (ton/ha)')
axes[0, 1].legend()

# Plot 3 — Yield distribution
data['Yield_ton_ha'].hist(ax=axes[1, 0], bins=20, color='#1A6B5F', edgecolor='white', alpha=0.85)
axes[1, 0].set_title('Yield Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Yield (ton/ha)')
axes[1, 0].set_ylabel('Frequency')

# Plot 4 — Correlation heatmap
num_cols = ['Rainfall_mm', 'Temperature_C', 'Nitrogen_kg_ha',
            'Phosphorus_kg_ha', 'Irrigation_times', 'Yield_ton_ha']
corr = data[num_cols].corr()
sns.heatmap(corr, ax=axes[1, 1], annot=True, fmt='.2f', cmap='YlGn',
            linewidths=0.5, annot_kws={'size': 8})
axes[1, 1].set_title('Feature Correlation Heatmap', fontweight='bold')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n [✓] EDA chart saved → eda_analysis.png")

# ──────────────────────────────────────────────
# 3. MACHINE LEARNING MODEL
# ──────────────────────────────────────────────
# Encode crop column
data['Crop_encoded'] = data['Crop'].map({'Wheat': 0, 'Maize': 1, 'Canola': 2})

features = ['Crop_encoded', 'Rainfall_mm', 'Temperature_C',
            'Nitrogen_kg_ha', 'Phosphorus_kg_ha',
            'Irrigation_times', 'Soil_pH', 'Sowing_density']

X = data[features]
y = data['Yield_ton_ha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train two models
lr  = LinearRegression()
rf  = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

print("\n" + "=" * 55)
print("   MODEL PERFORMANCE")
print("=" * 55)
print(f"\n Linear Regression:")
print(f"   R² Score : {r2_score(y_test, lr_pred):.3f}")
print(f"   MAE      : {mean_absolute_error(y_test, lr_pred):.3f} ton/ha")

print(f"\n Random Forest:")
print(f"   R² Score : {r2_score(y_test, rf_pred):.3f}")
print(f"   MAE      : {mean_absolute_error(y_test, rf_pred):.3f} ton/ha")

# ──────────────────────────────────────────────
# 4. FEATURE IMPORTANCE CHART
# ──────────────────────────────────────────────
importance = pd.Series(rf.feature_importances_, index=features).sort_values()

fig2, ax = plt.subplots(figsize=(9, 5))
importance.plot(kind='barh', color='#1A6B5F', edgecolor='white', ax=ax)
ax.set_title('Feature Importance — Random Forest Model\nBy: Sharaft Hussain',
             fontweight='bold', color='#1A6B5F')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n [✓] Feature importance chart saved → feature_importance.png")

# ──────────────────────────────────────────────
# 5. PREDICT NEW FIELD DATA
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("   PREDICT YIELD FOR NEW FIELD")
print("=" * 55)

new_field = pd.DataFrame([{
    'Crop_encoded':      0,       # 0=Wheat
    'Rainfall_mm':       350,
    'Temperature_C':     24,
    'Nitrogen_kg_ha':    120,
    'Phosphorus_kg_ha':  60,
    'Irrigation_times':  5,
    'Soil_pH':           7.2,
    'Sowing_density':    120,
}])

predicted = rf.predict(new_field)[0]
print(f"\n Input  : Wheat | Rainfall 350mm | Nitrogen 120 kg/ha")
print(f"          Temperature 24°C | Irrigations 5 times")
print(f"\n Predicted Yield : {predicted:.2f} ton/ha")
print(f"\n {'=' * 55}")
print("   Project Complete! — Sharaft Hussain")
print(f" {'=' * 55}\n")
