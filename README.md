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

Esto crea directorios como:

```
$ tree --sort=mtime dof | head -n 10
dof
├── 2025
│   ├── 01
│   │   ├── 02012025-MAT.pdf
│   │   ├── 03012025-MAT.pdf
│   │   ├── 06012025-MAT.pdf
│   │   ├── 07012025-MAT.pdf
...
```

Para convertirlos a formato markdown, ejecuta el siguiente comando:

Para convertir un folder completo:
```
marker --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  --skip_existing \
  --workers=1 \
  dof/2024/04/
# este comando tardó 2h 31m 23s en una macbook pro M3 de 36GB RAM
```

Para un archivo específico:
```
marker_single --output_dir dof_markdown/2024/04/ \
  --paginate_output \
  --languages="es" \
  dof/2024/04/01042024-MAT.pdf
# este comando tardó 2m 7s en una macbook pro M3 de 36GB RAM
# Sorprendentemente, porque otros pueden tardar más de 10 minutos.
```
