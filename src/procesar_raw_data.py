# %%
import logging
import os
from pathlib import Path

import yaml

from dr import obtener_funcion_ajuste_dr
from feature_engineering import (
    generar_variables_lag,
    imputar_lags_nulos_mediana,
    imputar_limites_por_deuda,
)
from utils import exportar_csv, formatear_periodos, importar_csv, separar_por_periodos

# Obtiene la ruta absoluta del directorio donde se encuentra el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Cambia el directorio de trabajo actual al del script
os.chdir(script_dir)

"""
# Debug: %cd "D:\OneDrive - Lisicki Litvin y Asociados\Maestria\Taller 1\modelo_scoring"
%cd "D:\OneDrive - Lisicki Litvin y Asociados\Maestria\Taller 1\modelo_scoring"
"""
# Cargar archivo YAML
ruta_cfg = Path("config/config.yaml")
with ruta_cfg.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Obtener la ruta al directorio de datos
raw_data_path = Path(cfg["paths"]["raw_data"])
processed_data_path = Path(cfg["paths"]["processed_data"])


# %%


def main():
    """
    Función principal para procesar los datos crudos.
    """
    # Construir la ruta completa al CSV
    data_cli_path = raw_data_path / "data_cli.csv"

    # Leer el CSV
    data_cli = importar_csv(data_cli_path)

    # Imputar valores nulos con 0
    for col in ["importe_pf", "importe_ca", "deuda_tc"]:
        data_cli[col] = data_cli[col].fillna(0)

    # Imputar valores nulos en los límites de tarjetas y cuentas según sus deudas asociadas
    data_cli = imputar_limites_por_deuda(
        data_cli, [("limite_tc", "deuda_tc"), ("limite_acc", "deuda_cc")]
    )

    # Inicio Estrategia Data Drifting
    tipo = cfg["params"]["dr"]["tipo_ajuste"]
    func_ajuste_dr = obtener_funcion_ajuste_dr(tipo)
    logging.info("Ejecutando ajuste de data drifting con el tipo: %s", tipo)
    data_cli = func_ajuste_dr(data_cli, cfg["params"])
    # Fin de la estrategia Data Drifting

    # Inicio Feature Engineering

    # Generar variables lag
    data_cli = generar_variables_lag(
        data_cli, cfg["params"]["fe"]["lags"], dropna_lags=False
    )
    # Fin Feature Engineering

    # Obtener peridos de train, validación y test
    df_train, df_val, df_test = separar_por_periodos(data_cli, cfg["params"]["ts"])

    periodos_train_unicos = sorted(df_train["periodo"].unique())
    periodos_val_unicos = sorted(df_val["periodo"].unique())
    periodos_test_unicos = sorted(df_test["periodo"].unique())

    logging.info(
        "Particiones iniciales — "
        "train: %d filas, periodos %s; "
        "val: %d filas, periodos %s; "
        "test: %d filas, periodos %s",
        len(df_train),
        formatear_periodos(periodos_train_unicos),
        len(df_val),
        formatear_periodos(periodos_val_unicos),
        len(df_test),
        formatear_periodos(periodos_test_unicos),
    )

    # Imputar nulos de las variables laggeadas
    df_train, df_val, df_test = imputar_lags_nulos_mediana(
        df_train,
        df_val,
        df_test,
        columnas=cfg["params"]["fe"]["lags"]["columnas"],
        n_lags=cfg["params"]["fe"]["lags"]["n_lags"],
    )

    logging.info(
        "Particiones tras imputación de lags — train: %d filas, val: %d, test: %d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    # Exportar los datasets procesados a CSV
    exportar_csv(df_train, processed_data_path / "train_data.csv")
    exportar_csv(df_val, processed_data_path / "val_data.csv")
    exportar_csv(df_test, processed_data_path / "test_data.csv")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Iniciando procesamiento de datos crudos...")
    # Llamar a la función principal
    main()

    logging.info("Procesamiento de datos crudos finalizado.")
    logging.info("Se han almacenado datasets procesados en en: %s", processed_data_path)
