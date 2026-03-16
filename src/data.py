from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """
    Devuelve la raíz del proyecto a partir de la ubicación de este archivo.
    """
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """
    Devuelve la carpeta data/raw del proyecto.
    """
    return get_project_root() / "data" / "raw"


def load_train() -> pd.DataFrame:
    """
    Carga el dataset de entrenamiento desde data/raw/train.csv.
    """
    train_path = get_data_dir() / "train.csv"
    return pd.read_csv(train_path)


def load_test() -> pd.DataFrame:
    """
    Carga el dataset de test desde data/raw/test.csv.
    """
    test_path = get_data_dir() / "test.csv"
    return pd.read_csv(test_path)