# Credit-Wise Loan Approval Analysis

Credit-Wise is a supervised machine-learning project that explores whether structured applicant and loan attributes can predict loan-approval outcomes. It demonstrates a complete classification workflow: missing-value treatment, exploratory analysis, categorical encoding, feature scaling, model comparison, feature engineering, and metric-based evaluation.

## Problem

Manual review of loan applications can be slow and inconsistent. This project investigates how historical application attributes—such as income, credit score, debt-to-income ratio, savings, collateral, employment, and requested loan terms—can be used to build a repeatable approval-classification baseline.

## Approach

The notebook:

1. Inspects a 1,000-row applicant dataset and handles missing values.
2. Explores class balance, applicant income, credit score, debt-to-income ratio, savings, and other approval signals.
3. Removes the applicant identifier from the feature set.
4. Encodes categorical variables with label and one-hot encoding.
5. Creates an 80/20 train-test split and standardizes features.
6. Compares Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes.
7. Adds squared credit-score and debt-to-income features and re-evaluates the models.

## Results

The initial comparison found that Logistic Regression and Naive Bayes both reached **86.5% accuracy**, with Naive Bayes producing the highest precision at **80.4%**. After feature engineering, Logistic Regression produced the strongest balanced result:

| Metric | Logistic Regression |
| --- | ---: |
| Accuracy | 88.0% |
| Precision | 78.5% |
| Recall | 83.6% |
| F1 score | 81.0% |

These results are an exploratory baseline from a single train-test split, not evidence that the model is ready for lending decisions.

## Technology

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes

## Repository contents

```text
Credit-Wise-Loan-Approval/
├── credit_wise.ipynb
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/yash807/Credit-Wise-Loan-Approval.git
cd Credit-Wise-Loan-Approval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook credit_wise.ipynb
```

Place the source dataset at the repository root as:

```text
loan_approval_data.csv
```

The expected data includes applicant income, co-applicant income, employment, age, marital status, dependents, credit score, existing loans, debt-to-income ratio, savings, collateral, loan amount and term, loan purpose, property area, education, gender, employer category, and the `Loan_Approved` target.

## Notebook viewing

The notebook outputs are intentionally cleared in Git to reduce file size and improve GitHub rendering. Run all cells locally to reproduce the exploratory charts and model metrics.

If GitHub's notebook preview is temporarily unavailable, open the notebook with [nbviewer](https://nbviewer.org/github/yash807/Credit-Wise-Loan-Approval/blob/main/credit_wise.ipynb).

## Responsible-use note

This repository is an educational data-science project and must not be used to approve or reject real loan applications. Production lending systems require representative data, fairness testing, explainability, regulatory review, human oversight, privacy controls, monitoring, and a formal appeal process. Sensitive or protected attributes should be reviewed carefully and excluded where required by law or policy.

## Limitations and next steps

- Results come from one dataset and one train-test split.
- Cross-validation and hyperparameter tuning have not yet been added.
- The project does not currently provide calibrated probabilities or a deployed inference service.
- Future work could add a reusable preprocessing pipeline, fairness metrics, ROC/PR analysis, model calibration, and an explainability layer.

