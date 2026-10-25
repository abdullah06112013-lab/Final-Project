# Telecom Customer Churn Prediction

An end-to-end machine learning project that predicts which telecom customers are likely to leave (churn), from a messy raw dataset to a saved model and an interactive prediction form.

## Project Goal

Telecom companies lose revenue when customers leave. This project cleans the raw customer data, finds the main reasons behind churn, and trains models to predict churn early so the company can act.

## Dataset

- File: `telecom_churn_raw.csv`
- 1,300 rows and 23 columns (customer details, services, billing, support calls, satisfaction and churn)
- After cleaning: 1,239 rows (`telecom_churn_clean.csv`)
- About 33% of customers churned

## Data Problems Found

- Hidden missing values written as `--`, `Not Available`, `?` and blanks, plus real NaN values
- 35 exact duplicate rows, near-duplicates (spaces and letter case) and repeated `customer_id`
- Inconsistent text in gender, yes/no columns, churn, contract type, payment method and customer satisfaction
- Numbers stored as text (`$3,869.1`, `101.56 USD`)
- Impossible values (negative tenure, charges and calls) and extreme values (999 months, 9999 GB)
- Mixed date formats and impossible dates in `signup_date`

## Cleaning Steps

1. Turned hidden missing values into NaN
2. Stripped text, made it lowercase and merged spelling variants
3. Converted numbers and dates to proper types
4. Removed duplicates after standardizing (1,300 to 1,251 rows)
5. Set impossible values and extreme values (above Q3 + 3 x IQR) to NaN
6. Filled missing numbers with the median and missing categories with the mode; dropped rows with missing churn

The raw file was left untouched.

## Key Findings

| Chart | Insight |
|---|---|
| Churn count | 33% of customers churned, so the data is imbalanced |
| Churn by contract type | Month-to-month customers churn the most (about 50%), two-year the least (about 6%) |
| Tenure by churn | Churned customers have a median tenure of 19 months vs 26 months for the others |
| Churn by internet service | Fiber optic users churn the most (about 41%), customers with no internet the least (about 18%) |
| Monthly charges by churn | Churned customers pay about 80 per month on average vs 70 for the others |

![Churn count](screenshots/churn_count.png)
![Churn by contract](screenshots/churn_by_contract.png)
![Tenure by churn](screenshots/tenure_by_churn.png)
![Churn by internet service](screenshots/churn_by_internet.png)
![Monthly charges by churn](screenshots/monthly_charges.png)

## Feature Engineering

- `avg_spend_per_month` = total_charges / (tenure_months + 1)
- `new_customer` = 1 if tenure is 12 months or less, otherwise 0

## Modelling

- Categorical variables were one-hot encoded and numeric features were scaled with `StandardScaler`
- Both steps are fitted only on the training data inside a pipeline, so there is no data leakage
- 80/20 train-test split with stratification
- Four baseline models: Logistic Regression, Decision Tree, Random Forest and XGBoost
- Random Forest and XGBoost were tuned with `RandomizedSearchCV` (5 iterations, 3-fold cross-validation, scored on F1)

### Confusion Matrices (tuned models)

![Confusion matrices](screenshots/confusion_matrices.png)

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression** | **0.730** | **0.609** | **0.481** | **0.538** |
| Random Forest | 0.718 | 0.587 | 0.457 | 0.514 |
| XGBoost | 0.718 | 0.596 | 0.420 | 0.493 |
| Tuned Random Forest | 0.710 | 0.576 | 0.420 | 0.486 |
| Decision Tree | 0.673 | 0.500 | 0.469 | 0.484 |
| Tuned XGBoost | 0.718 | 0.608 | 0.383 | 0.470 |

## Final Model: Logistic Regression

Logistic Regression scored highest on all four metrics (F1 0.538, recall 0.481), and the tuned models did not beat it. It is also simple and fast, and it gives a clear churn probability for the prediction form.

The final model and its preprocessing steps are saved together with `joblib` in `churn_model.pkl`.

## Prediction Form

The notebook ends with an interactive form (Google Colab) where you enter customer details and get the churn prediction and churn probability.

Inputs: gender, partner, tenure months, monthly charges, total charges, internet service, payment method, contract type.

![Prediction form](screenshots/prediction_form.png)

## How to Run

1. Open `final_project_fixed.ipynb` in Google Colab.
2. Upload `telecom_churn_raw.csv` to the `/content/` folder.
3. Run all cells from top to bottom.
4. Use the form in the last cell to test a customer.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib.

## Project Files

```
telecom-churn-prediction/
├── final_project_fixed.ipynb
├── telecom_churn_raw.csv
├── telecom_churn_clean.csv
├── churn_model.pkl
├── Telecom-Customer-Churn-Prediction.pptx
├── screenshots/
└── README.md
```
