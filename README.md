# Calories Burnt Prediction

Machine learning regression pipeline comparing five algorithms to predict exercise calorie expenditure across 15,000 workout sessions, achieving R² = 0.947 with XGBoost.

---

## Project Overview

Fitness trackers estimate calorie burn — but how accurate are they, and which features actually drive the prediction? This project builds and compares five regression models to predict calories burned during exercise based on physiological and activity data. The goal is to find the most accurate and generalisable model, and to surface the overfitting risks that arise when choosing between tree-based approaches.

With XGBoost predicting within ~10 calories on average (R² = 0.947), the model is accurate enough to power real-time feedback loops in wearables applications or personalised health coaching platforms.

---

## Dataset

- **Source:** Combined exercise + calories datasets (Kaggle)
- **Size:** ~15,000 workout sessions
- **Files:** `exercise.csv` (activity and physiological features), `calories.csv` (calorie labels)

| Feature | Description |
|---------|-------------|
| Age, Gender | Demographic features |
| Height, Weight | Body composition |
| Duration | Session length in minutes |
| Heart_Rate | Average heart rate during session |
| Body_Temp | Body temperature during exercise |
| Calories | Target variable — calories burnt |

<p align="center">
  <img src="images/dataset_merge.png" alt="Dataset Merge" width="600"/>
</p>

---

## Approach

1. **Data loading & merging** — combined `exercise.csv` and `calories.csv` on session ID
2. **EDA** — checked for missing values and duplicates; correlation analysis between features and calorie burn; visualised distributions across activities and demographics
3. **Model training** — trained five regression models:
   - Linear Regression
   - Lasso Regression
   - Ridge Regression
   - Random Forest Regressor
   - XGBoost Regressor
4. **Evaluation** — compared using MAE, MSE, and R² on both train and test sets to detect overfitting

---

## Key Findings

| Model | Train R² | Test R² | Test MAE |
|-------|----------|---------|----------|
| Linear Regression | 0.870 | 0.870 | 17.99 |
| Lasso | 0.869 | 0.869 | 18.01 |
| Ridge | 0.870 | 0.870 | 17.99 |
| Random Forest | **0.992** | 0.942 | 10.67 |
| **XGBoost** | 0.970 | **0.947** | **10.33** |

<p align="center">
  <img src="images/model_comparison.png" alt="Model Comparison" width="800"/>
</p>

- **XGBoost is the best overall model** — Test R² = 0.947, MAE ≈ 10.33 calories. Best balance of accuracy and generalisability
- **Random Forest shows clear overfitting** — train R² = 0.992 vs test R² = 0.942; the gap signals the model has memorised training patterns rather than generalising
- **Linear models underfit** — R² ≈ 0.87 across Linear, Lasso, and Ridge, indicating non-linear relationships in the features that trees capture and linear models cannot
- **Session duration and heart rate** are the strongest predictors of calorie expenditure — body composition features are secondary
- An MAE of ~10 calories is accurate enough for real-time feedback in wearables or health coaching platforms

<p align="center">
  <img src="images/workflow.png" alt="Project Workflow" width="600"/>
</p>

---

## How to Run

```bash
git clone https://github.com/SamadZaheer/Calories-Burnt-Prediction.git
cd Calories-Burnt-Prediction
pip install -r requirements.txt
jupyter notebook "Calories Burnt Prediction.ipynb"
```

---

## Author

**Samad Zaheer** — Master of Information Technology (Data Science), Queensland University of Technology (QUT)
