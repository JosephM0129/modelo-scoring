import logging
from typing import Dict, List, Tuple

import pandas as pd


def generar_variables_lag(
    df: pd.DataFrame, cfg: Dict, dropna_lags: bool = False
) -> pd.DataFrame:
    """
    Genera variables lag de 1 a n_lags para las columnas indicadas,
    solo si cfg['params']['fe']['lags']['habilitar'] es True.
    """

    if not cfg.get("habilitar", False):
        logging.info("Generación de lags desactivada en configuración.")
        return df

    columnas = cfg["columnas"]
    n_lags = cfg["n_lags"]

    if n_lags <= 0:
        logging.info("n_lags <= 0, no se generarán variables lag.")
        return df

    df_lags = df.sort_values(by="periodo").copy()
    for col in columnas:
        if col not in df_lags.columns:
            logging.warning("No existe la columna '%s'; se omite.", col)
            continue
        for i in range(1, n_lags + 1):
            nombre_lag = f"{col}_lag_{i}"
            df_lags[nombre_lag] = df_lags[col].shift(i)

    if dropna_lags:
        lag_cols = [f"{col}_lag_{i}" for col in columnas for i in range(1, n_lags + 1)]
        df_lags = df_lags.dropna(subset=lag_cols).reset_index(drop=True)

    return df_lags


def imputar_limites_por_deuda(
    df: pd.DataFrame, pares: List[Tuple[str, str]]
) -> pd.DataFrame:
    """
    Imputa valores nulos en columnas de límite según su deuda asociada,
    y luego rellena el resto de nulos con 0.

    Args:
        df (pd.DataFrame): DataFrame original.
        pares (List[Tuple[str, str]]): Lista de tuplas (col_limite, col_deuda).
            Ejemplo: [("limite_tc", "deuda_tc"), ("limite_acc", "deuda_cc")]

    Returns:
        pd.DataFrame: Copia del DataFrame con los límites imputados.
    """
    df_imp = df.copy()

    for col_limite, col_deuda in pares:
        if col_limite not in df_imp.columns or col_deuda not in df_imp.columns:
            raise KeyError(f"Columnas no encontradas: {col_limite}, {col_deuda}")

        # 1) Si límite es NaN y deuda NO, copiar deuda
        mask = df_imp[col_limite].isna() & df_imp[col_deuda].notna()
        df_imp.loc[mask, col_limite] = df_imp.loc[mask, col_deuda]

        # 2) Rellenar cualquier NaN restante en límite con 0
        df_imp[col_limite] = df_imp[col_limite].fillna(0)

    return df_imp


def imputar_lags_nulos_mediana(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    columnas: List[str],
    n_lags: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Imputa los NaN de las variables lag (de 1..n_lags) usando la mediana
    calculada únicamente sobre df_train, y aplica ese mismo valor en val y test.

    Args:
        df_train, df_val, df_test: particiones del dataset.
        columnas: lista de columnas originales sobre las que se crearon lags.
        n_lags: número de lags generados.

    Returns:
        Tupla (df_train_imp, df_val_imp, df_test_imp) con los NaNs rellenados.
    """
    # 1) Construir lista de nombres de columnas lag
    lag_cols = [f"{col}_lag_{i}" for col in columnas for i in range(1, n_lags + 1)]

    # 2) Calcular medianas sobre df_train
    medianas = df_train[lag_cols].median(skipna=True)

    # 3) Rellenar NaNs en cada partición con esas medianas
    for df in (df_train, df_val, df_test):
        # reindexar para mantener las columnas lag aunque alguna no exista
        df[lag_cols] = df[lag_cols].fillna(medianas)

    return df_train, df_val, df_test
