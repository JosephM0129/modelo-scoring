# %%
import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from utils import importar_excel


def importar_ipc(ipc_excel: Union[str, Path], ipc_hoja: str) -> pd.DataFrame:
    """
    Importa el índice IPC desde un archivo Excel y lo procesa.

    Args:
        ipc_excel (Union[str, Path]): Ruta al archivo Excel que contiene el IPC.
        ipc_hoja (str): Nombre de la hoja del Excel que contiene los datos del IPC.

    Returns:
        pd.DataFrame: DataFrame con el índice IPC procesado, con índice AAAAMM (int)
                      y única columna "Nivel general" de tipo float.

    Raises:
        ValueError: Si no se encuentra la fila "Nivel general", si hay nulos
                    en esa columna, o si falla la conversión a float.

    Fuente: https://www.indec.gob.ar/indec/web/Nivel4-Tema-3-5-31

    Tema: Series históricas | Índice de precios al consumidor con cobertura nacional

    Archivo: Índices y variaciones porcentuales mensuales e interanuales
    """
    # 1) Leer Excel con tu función importar_excel (wrapper de pd.read_excel)
    logging.debug(
        "Iniciando la importación de IPC desde %s, hoja %s", ipc_excel, ipc_hoja
    )

    tot_nacional_header = 5
    df_ipc = importar_excel(ipc_excel, sheet_name=ipc_hoja, header=tot_nacional_header)

    # 2) Filtrar la fila de "Nivel general"
    mask = df_ipc["Total nacional"] == "Nivel general"
    if not mask.any():
        raise ValueError(
            "No se encontró la fila 'Nivel general' en la columna 'Total nacional'."
        )
    ipd_ng = df_ipc[mask].reset_index(drop=True).iloc[[0]]

    # 3) Transponer y reestructurar
    ipd_ng = ipd_ng.T
    ipd_ng.columns = ipd_ng.iloc[0]  # la primera fila pasa a ser nombres de columna
    ipd_ng = ipd_ng.drop(index=ipd_ng.index[0])  # eliminar la fila redundante
    # conviertes el índice a AAAAMM int (ya lo tienes)
    ipd_ng.index = (
        pd.to_datetime(ipd_ng.index, format="%Y-%m").strftime("%Y%m").astype(int)
    )

    # renombras el nombre del índice
    ipd_ng.index.name = "Total nacional"

    # reset_index() lo convierte en columna y restaura un índice 0,1,2…
    ipd_ng = ipd_ng.reset_index()

    ipd_ng["Total nacional"] = ipd_ng["Total nacional"].astype(int)

    # 4) Validar nulos en la columna "Nivel general"
    columna = "Nivel general"
    if ipd_ng[columna].isnull().any():
        raise ValueError(f"Se detectaron valores nulos en la columna '{columna}'.")

    # 5) Intentar convertir a float y atrapar errores
    try:
        ipd_ng[columna] = ipd_ng[columna].astype(float)
    except Exception as e:
        raise ValueError(f"Error al convertir la columna '{columna}' a float: {e}")

    logging.info(
        "IPC importado y procesado correctamente desde %s, hoja %s", ipc_excel, ipc_hoja
    )

    return ipd_ng


def ajuste_dr_ipc(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Incorpora el IPC de la hoja dada, verifica
    integridad y ajusta las columnas indicadas por data drifting (inflación).

    Args:
        df: datos de cliente.
        ipc_excel (Union[str, Path]): Ruta al archivo Excel con el IPC.
        ipc_hoja (str): Nombre de la hoja en el Excel que contiene el IPC.
        columnas_dr (List[str]): Lista de nombres de columnas de data_cli
                                 que deben ajustarse por IPC.

    Returns:
        pd.DataFrame: DataFrame resultante con las columnas ajustadas.

    Raises:
        ValueError: Si faltan valores nulos en “Nivel general” o falla la conversión.
    """

    # 2) Importar IPC y merge
    ipc = importar_ipc(cfg["dr"]["ipc"]["archivo"], cfg["dr"]["ipc"]["hoja"])
    df = df.merge(ipc, left_on="periodo", right_on="Total nacional", how="left")

    # 3) Validar no-nulos en "Nivel general"
    col_ng = "Nivel general"
    if df[col_ng].isnull().any():
        raise ValueError(
            f"Se detectaron valores nulos en la columna '{col_ng}' tras el merge."
        )

    # 4) Calcular índice base (mes más reciente)
    mes_base = df["periodo"].max()
    indice_base = df.loc[df["periodo"] == mes_base, col_ng].iloc[0]
    df["indice_base"] = indice_base

    # 5) Ajustar columnas de drifting
    columnas_dr = cfg["dr"]["columnas"]
    for col in columnas_dr:
        if col in df.columns:
            try:
                factor = (df["indice_base"] / df[col_ng]).astype(float)
                df[col] = df[col] * factor.fillna(1)
            except Exception as e:
                raise ValueError(f"Error ajustando la columna '{col}': {e}")
        else:
            logging.warning(
                "La columna '%s' no existe en el DataFrame y se omite.", col
            )

    # 6) Limpiar columnas auxiliares
    df = df.drop(columns=["Total nacional", "indice_base", "Nivel general"])

    return df
