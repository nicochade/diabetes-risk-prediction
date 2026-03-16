from __future__ import annotations

import pandas as pd


GENDER_MAP = {
    "Female": 0,
    "Male": 1,
    "Other": 2,
}

SMOKING_HISTORY_MAP = {
    "No Info": 0,
    "current": 1,
    "ever": 2,
    "former": 3,
    "never": 4,
    "not current": 5,
}

BMI_CATEGORY_MAP = {
    "Underweight": 0,
    "Normal": 1,
    "Overweight": 2,
    "Obese": 3,
}


def categorize_bmi(bmi: float) -> str:
    """
    Asigna una categoría de BMI según rangos estándar.
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones base que deben usarse tanto en train como en test.

    Incluye:
    - codificación de gender
    - codificación de smoking_history
    - creación y codificación de bmi_category
    """
    df = df.copy()

    df["gender"] = df["gender"].map(GENDER_MAP)

    df["smoking_history"] = df["smoking_history"].astype(str).map(SMOKING_HISTORY_MAP)

    df["bmi_category"] = df["bmi"].map(categorize_bmi).map(BMI_CATEGORY_MAP)

    return df


def apply_train_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros derivados del EDA solo sobre el dataset de entrenamiento.
    """
    df = df.copy()

    df = df[df["age"] < 80]
    df = df[df["bmi"] != 27.32]

    return df


def prepare_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset de entrenamiento replicando la lógica del notebook:
    transformación base + filtros derivados del EDA.
    """
    df = clean_base(df)
    df = apply_train_filters(df)
    return df


def prepare_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara el dataset de test aplicando solo transformaciones base.
    No aplica filtros derivados del EDA.
    """
    df = clean_base(df)
    return df