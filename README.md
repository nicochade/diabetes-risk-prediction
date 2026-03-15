# Diabetes Risk Prediction

Machine learning project to predict diabetes risk from demographic and clinical features, with a recall-oriented evaluation strategy for medical screening settings.

## Project overview

This project explores a supervised learning approach to predict diabetes risk using tabular patient data.  
The modeling strategy prioritizes **recall**, since in a screening context false negatives are usually more costly than false positives.

The project started from an academic competition notebook and is being refactored into a more reproducible and portfolio-ready machine learning repository.

## Dataset

The dataset contains approximately:

- 95,000 rows in the training set
- 5,000 rows in the test set
- 10 input features
- 8.5% positive class prevalence

Features include demographic and clinical variables such as:

- age
- gender
- hypertension
- heart disease
- smoking history
- BMI
- HbA1c level
- blood glucose level

## Objective

Build and evaluate models for diabetes risk prediction with emphasis on:

- high recall
- class imbalance awareness
- threshold-dependent evaluation
- clinical interpretation of false negatives vs false positives

## Current repository status

This repository is under active refactoring.  
The original competition notebook is preserved in `notebooks/`, and the project is being restructured into modular Python code under `src/`.

Planned improvements include:

- reproducible preprocessing pipeline
- fair comparison of baseline and boosted models
- threshold tuning
- precision-recall analysis
- final feature importance interpretation

## Repository structure

```text
diabetes-risk-prediction/
├── data/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
