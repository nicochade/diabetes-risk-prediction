import pandas as pd


def apply_eda_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply row-level filters derived from exploratory data analysis.

    Notes
    -----
    These filters are kept separate from the base preprocessing pipeline
    because they come from dataset-specific EDA decisions and may require
    additional justification in the final project narrative.
    """
    df = df.copy()
    df = df[df["age"] < 80]
    df = df[df["bmi"] != 27.32]
    return df


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create BMI category feature from bmi values.
    """
    df = df.copy()

    bins = [0, 18.5, 24.9, 29.9, 100]
    labels = ["underweight", "normal", "overweight", "obese"]

    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using the same mapping strategy
    used in the original notebook.
    """
    df = df.copy()

    gender_map = {
        "Female": 0,
        "Male": 1,
        "Other": 2,
    }

    smoking_map = {
        "never": 0,
        "No Info": 1,
        "former": 2,
        "current": 3,
        "not current": 4,
        "ever": 5,
    }

    bmi_cat_map = {
        "underweight": 0,
        "normal": 1,
        "overweight": 2,
        "obese": 3,
    }

    if "gender" in df.columns:
        df["gender"] = df["gender"].map(gender_map)

    if "smoking_history" in df.columns:
        df["smoking_history"] = df["smoking_history"].map(smoking_map)

    if "bmi_category" in df.columns:
        df["bmi_category"] = df["bmi_category"].astype(str).map(bmi_cat_map)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns not used for modeling in the original notebook.
    """
    df = df.copy()

    cols_to_drop = [col for col in ["patient", "bmi"] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply base preprocessing steps shared across datasets.
    """
    df = df.copy()
    df = add_bmi_category(df)
    df = encode_features(df)
    df = drop_unused_columns(df)
    return df


def preprocess_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess training data following the original notebook logic:
    EDA-based filters + base preprocessing.
    """
    df = df.copy()
    df = apply_eda_filters(df)
    df = basic_cleaning(df)
    return df


def preprocess_test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess test data following the original notebook logic:
    base preprocessing without row filtering.
    """
    df = df.copy()
    df = basic_cleaning(df)
    return df