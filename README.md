# dof-rag
dog-raf es un chat y un sistema de consulta por generación aumentada para explorar las ediciones del Diario Oficial de la Federación de México.

# Requerimientos

Instala [uv](https://docs.astral.sh/uv/) para manejar las dependencias y ejecutar el proyecto.

Una vez instalado uv, ejecuta el siguiente comando para iniciar el proyecto:

```
uv venv # Crear un entorno virtual
uv sync # Sincronizar dependencias
uv run get_dof.py --help
uv run get_dof.py --start-year=2025 --end-year=2023
```
