# encoding: utf-8

# Description: Script para descargar el Diario Oficial de la Federación (DOF) de México

#wget --https-only "https://diariooficial.gob.mx/abrirPDF.php?archivo=23012025-MAT.pdf&anio=2025&repo=" -O "DOF_23012025-MAT.pdf"

import os
import requests
import ssl
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager
from datetime import datetime

class TLSAdapter(HTTPAdapter):
    """Adaptador personalizado para forzar TLS 1.2 o 1.3"""
    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context or ssl.create_default_context()
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self.ssl_context
        return super().init_poolmanager(*args, **kwargs)

# Crear un contexto SSL con TLS 1.2 y menos restricciones
ssl_context = ssl.create_default_context()
ssl_context.set_ciphers("DEFAULT@SECLEVEL=1")  # Reduce nivel de seguridad si es necesario

session = requests.Session()
session.mount("https://", TLSAdapter(ssl_context=ssl_context))

def get_url(year, month, day):
    base_url = "https://diariooficial.gob.mx/abrirPDF.php?archivo="
    return f"{base_url}{day}{month}{year}-MAT.pdf&anio={year}&repo="

def get_dof(year, month, day):
    url = get_url(year, month, day)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = session.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Error {response.status_code}: No se pudo descargar {url}")


def save_dof(year, month, day):
    """Descarga y guarda el PDF solo si no está vacío."""
    content = get_dof(year, month, day)

    if not content or len(content) < 1000:  # Umbral de 1000 bytes para evitar archivos vacíos o inválidos
        print(f"⚠️ Archivo {day}{month}{year}-MAT.pdf está vacío o no válido. No se guardará.")
        return
    
    folder = f"./dof/{year}/{month}"
    filename = f"{day}{month}{year}-MAT.pdf"
    filepath = os.path.join(folder, filename)

    os.makedirs(folder, exist_ok=True)  # Crea la carpeta si no existe

    with open(filepath, "wb") as file:
        file.write(content)

    print(f"✅ Archivo guardado: {filepath}")


# Guardar los PDFs
mes_dias = {
    "01": 31, "02": 28, "03": 31, "04": 30, "05": 31, "06": 30,
    "07": 31, "08": 31, "09": 30, "10": 31, "11": 30, "12": 31
}

_break = False
for month, days in mes_dias.items():
    if _break:
        break
    for day in range(1, days + 1):
        # break if day is gretter than today
        _month = datetime.now().strftime("%m")
        _day = datetime.now().strftime("%d")
        if int(month) >= int(_month) and int(day) > int(_day):
            _break = True
            break
        save_dof("2025", month, f"{day:02d}")  # f"{day:02d}" -> 01, 02


