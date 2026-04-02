# 🪞 Mind Mirror — Depression Risk Screening Tool

**CS 7180 Special Topics in AI — Spring 2026**  
Andrew Min & Mengyi Ma

A Streamlit app that predicts depression risk from a 2-minute personality survey (TIPI) and basic demographics, using a Gradient Boosting classifier trained on 39,775 DASS-42 survey responses.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Structure

```
mind_mirror/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
└── data/
    ├── mental_health_model.pkl   # Trained Gradient Boosting model
    ├── feature_cols.pkl          # Feature column names
    ├── cv_results.json           # Cross-validation results (6 models)
    ├── confusion_matrix.json     # Test set confusion matrix
    ├── roc_data.json             # ROC curve data points
    ├── feature_importance.json   # Feature importance scores
    ├── classification_report.json # Full classification report
    ├── country_risk.csv          # Country-level risk rates
    ├── age_stats.csv             # Age group risk statistics
    ├── tipi_profiles.csv         # TIPI personality profiles
    └── dataset_avg.json          # Overall dataset average risk
```

## Three Tabs

1. **Screener** — Interactive form with TIPI sliders + demographics → risk prediction with probability, top feature drivers, and plain-language explanation.
2. **Global Insights** — Country risk rates, age group breakdown, and personality profile comparisons between at-risk and low-risk groups.
3. **Model Card** — Cross-validation table, confusion matrix, ROC curve, feature importance, threshold rationale, and known limitations.

## Model Details

- **Algorithm:** Gradient Boosting (n_estimators=200, max_depth=3, learning_rate=0.05)
- **Threshold:** DASS-42 depression score ≥ 21 (severe+)
- **Test Accuracy:** 82% | **F1 (At-Risk):** 0.89 | **AUC:** 0.84
- **Features:** 10 TIPI personality scores + 9 demographic variables
