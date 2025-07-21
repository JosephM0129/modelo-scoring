# Patrón Estrategia (Estrategy Pattern) para ajustes de data drifting
from typing import Callable, Dict

from dr.ipc import ajuste_dr_ipc

AJUSTES: Dict[str, Callable] = {
    "ipc": ajuste_dr_ipc,
    # "uva": ajuste_dr_uva,
    # 'tc_mep': ajuste_dr_tc_mep,
}


def obtener_funcion_ajuste_dr(tipo: str) -> Callable:
    try:
        return AJUSTES[tipo]
    except KeyError:
        raise ValueError(
            f"Tipo de ajuste desconocido: '{tipo}'. Opciones válidas: {list(AJUSTES)}"
        )
