# %%
import logging
import re
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class FiltroPeriodo:
    """
    Permite parsear una cadena de periodos en formato 'AAAAMM' o rangos 'AAAAMM-AAAAMM',
    devolviendo la lista de periodos individuales inclusive.
    """

    _RE_PERIODO = re.compile(r"^(?P<anio>\d{4})(?P<mes>0[1-9]|1[0-2])$")

    def __init__(self, cadena_periodos: str) -> None:
        """
        Args:
            cadena_periodos (str): Periodos separados por comas, p.ej. "202501,202503-202506"
        """
        # Elimina espacios y guarda la cadena limpia
        self.cadena = cadena_periodos.replace(" ", "")
        logger.debug("Inicializado FiltroPeriodo con '%s'", self.cadena)

    @classmethod
    def _validar_formato(cls, periodo: str) -> None:
        """
        Verifica que 'periodo' cumpla el formato AAAAMM y mes entre 01 y 12.

        Args:
            periodo (str): Cadena a validar.

        Raises:
            ValueError: Si el formato o el mes son inválidos.
        """
        if not cls._RE_PERIODO.match(periodo):
            raise ValueError(
                f"Período inválido '{periodo}': debe ser AAAAMM y mes entre 01–12."
            )

    def _expandir_rango(self, inicio: str, fin: str) -> List[str]:
        """
        Expande el rango desde 'inicio' hasta 'fin', inclusive.

        Args:
            inicio (str): Primer período (AAAAMM).
            fin (str): Último período (AAAAMM).

        Returns:
            List[str]: Lista de periodos AAAAMM.

        Raises:
            ValueError: Si el rango está invertido o algún formato es inválido.
        """
        self._validar_formato(inicio)
        self._validar_formato(fin)

        fecha_i = datetime.strptime(inicio, "%Y%m")
        fecha_f = datetime.strptime(fin, "%Y%m")
        if fecha_i > fecha_f:
            raise ValueError(f"Rango inválido: '{inicio}' > '{fin}'.")

        periodos: List[str] = []
        actual = fecha_i
        while actual <= fecha_f:
            periodos.append(actual.strftime("%Y%m"))
            # Avanza un mes
            month = actual.month + 1
            year = actual.year + (month > 12)
            month = month if month <= 12 else 1
            actual = datetime(year, month, 1)
        logger.debug("Rango %s-%s expandido a %s", inicio, fin, periodos)
        return periodos

    def obtener_periodos(self) -> List[int]:
        """
        Parsea la cadena de entrada y retorna todos los periodos individuales como enteros AAAAMM.

        Returns:
            List[int]: Lista de periodos en orden.

        Example:
            >>> FiltroPeriodo("202501, 202503-202505").obtener_periodos()
            [202501, 202503, 202504, 202505]
        """
        # Paso 1: recolectar todos los tokens como strings
        resultado: List[str] = []
        for token in self.cadena.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                inicio, fin = map(str.strip, token.split("-", 1))
                resultado.extend(self._expandir_rango(inicio, fin))
            else:
                self._validar_formato(token)
                resultado.append(token)

        logger.debug("Períodos finales (strings): %s", resultado)

        # Paso 2: eliminar duplicados y ordenar
        resultado_unicos = sorted(set(resultado))

        # Paso 3: convertir cada período AAAAMM de str a int
        periodos_int = [int(p) for p in resultado_unicos]
        logger.debug("Períodos finales (enteros): %s", periodos_int)

        return periodos_int


def importar_csv(path: str, delimiter: str = ";") -> pd.DataFrame:
    """
    Importa un archivo CSV y lo devuelve como DataFrame de pandas.

    Args:
        path (str): Ruta al archivo CSV.
        delimiter (str): Delimitador del CSV. Por defecto es ';'.

    Returns:
        pd.DataFrame: DataFrame con los datos del CSV.
    """
    try:
        df = pd.read_csv(path, delimiter=delimiter)
        logger.info("CSV importado correctamente desde %s", path)
        return df
    except Exception as e:
        logger.error("Error al importar CSV desde %s: %s", path, e)
        raise


def importar_excel(path: str, sheet_name: str = "Sheet1", header=0) -> pd.DataFrame:
    """
    Importa un archivo Excel y lo devuelve como DataFrame de pandas.

    Args:
        path (str): Ruta al archivo Excel.
        sheet_name (str): Nombre de la hoja a importar. Por defecto es 'Sheet1'.

    Returns:
        pd.DataFrame: DataFrame con los datos del Excel.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, header=header)
        logger.info("Excel importado correctamente desde %s", path)
        return df
    except Exception as e:
        logger.error("Error al importar Excel desde %s: %s", path, e)
        raise


def exportar_csv(df: pd.DataFrame, path: str, delimiter: str = ";") -> None:
    """
    Exporta un DataFrame a un archivo CSV

    Args:
        df (pd.DataFrame): DataFrame a exportar.
        path (str): Ruta donde guardar el archivo CSV.
        delimiter (str): Delimitador del CSV. Por defecto es ';'.
    """
    try:
        df.to_csv(path, sep=delimiter, index=False)
        logger.info("DataFrame exportado correctamente a %s", path)
    except Exception as e:
        logger.error("Error al exportar DataFrame a %s: %s", path, e)
        raise


def separar_por_periodos(
    df: pd.DataFrame, definiciones: Dict[str, str], columna_periodo: str = "periodo"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separa el DataFrame en train, val y test según los rangos de periodos
    dados en el dict definiciones = {
        "train": "202501-202512",
        "val":   "202601-202602",
        "test":  ["202603", "202604-202605"]
    }.
    """
    particiones: Dict[str, pd.DataFrame] = {}
    for clave, definicion in definiciones.items():
        periodos = FiltroPeriodo(definicion).obtener_periodos()
        df_sub = df[df[columna_periodo].isin(periodos)].copy()
        if df_sub.empty:
            raise ValueError(f"Partición '{clave}' quedó vacía para: {periodos}")
        particiones[clave] = df_sub
    return particiones["train"], particiones["val"], particiones["test"]


def formatear_periodos(periodos):
    """
    Si periodos = [a], devuelve "a".
    Si periodos = [a, ..., b], devuelve "a-b".
    """
    if not periodos:
        return ""
    primero = periodos[0]
    ultimo = periodos[-1]
    return str(primero) if primero == ultimo else f"{primero}-{ultimo}"
