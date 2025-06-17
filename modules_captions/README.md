# Módulo de Extracción de Descripciones de Imágenes

Sistema modular para generar descripciones automáticas de imágenes utilizando IA, con almacenamiento en SQLite y soporte para múltiples proveedores.

## Características

- **Base de datos SQLite**: Almacenamiento transaccional con operaciones CRUD
- **Múltiples proveedores de IA**: OpenAI, Gemini, Claude, Ollama, Azure OpenAI
- **Arquitectura modular**: Componentes especializados y reutilizables
- **Manejo de errores**: Sistema centralizado con logging estructurado
- **Procesamiento por lotes**: Optimización de rendimiento con checkpoints
- **Configuración flexible**: CLI, archivos JSON y variables de entorno

## Instalación

```bash
pip install openai pillow python-dotenv colorama tqdm
```

## Configuración

### Variables de Entorno

```bash
export OPENAI_API_KEY="tu_clave_api"        # OpenAI
export GEMINI_API_KEY="tu_clave_api"        # Google Gemini  
export ANTHROPIC_API_KEY="tu_clave_api"     # Anthropic Claude
export AZURE_OPENAI_API_KEY="tu_clave_api"  # Azure OpenAI
```

### Estructura del Proyecto

```
modules_captions/
├── extract_captions.py      # Script principal
├── config.json              # Configuración de proveedores
├── clients/                 # Clientes de IA
├── db/                      # Gestión de base de datos
├── utils/                   # Utilidades principales
└── logs/                    # Archivos de log
```

## Proveedores Soportados

| Proveedor | Modelos Principales | Variable de Entorno |
|-----------|-------------------|--------------------|
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| **Gemini** | gemini-2.0-flash, gemini-1.5-pro | `GEMINI_API_KEY` |
| **Claude** | claude-sonnet-4, claude-3-5-sonnet | `ANTHROPIC_API_KEY` |
| **Ollama** | llama3.2-vision (local) | No requerida |
| **Azure OpenAI** | gpt-4-vision | `AZURE_OPENAI_API_KEY` |

## Uso

### Comandos Básicos

```bash
# Usar proveedor por defecto (Gemini)
python extract_captions.py --root-dir ./imagenes

# Especificar proveedor
python extract_captions.py --root-dir ./imagenes --openai
python extract_captions.py --root-dir ./imagenes --gemini
python extract_captions.py --root-dir ./imagenes --claude

# Forzar reprocesamiento de imágenes existentes
python extract_captions.py --root-dir ./imagenes --force-reprocess

# Ver estado del sistema
python extract_captions.py --status
```

### Parámetros Disponibles

| Parámetro | Descripción | Ejemplo |
|-----------|-------------|----------|
| `--root-dir` | Directorio con imágenes | `--root-dir ./imagenes` |
| `--db-path` | Ruta de base de datos | `--db-path captions.db` |
| `--openai/--gemini/--claude` | Seleccionar proveedor | `--gemini` |
| `--force-reprocess` | Reprocesar imágenes existentes | `--force-reprocess` |
| `--status` | Mostrar estado del sistema | `--status` |
| `--log-level` | Nivel de logging | `--log-level DEBUG` |
| `--debug` | Modo depuración detallado | `--debug` |

### Configuración Personalizada

El archivo `config.json` permite personalizar el comportamiento del sistema:

```json
{
  "provider": "gemini",
  "root_directory": "./imagenes",
  "db_path": "captions.db",
  "commit_interval": 10,
  "log_level": 20,
  "providers": {
    "gemini": {
      "client_config": {
        "model": "gemini-2.0-flash",
        "max_tokens": 256,
        "temperature": 0.6
      }
    }
  }
}
```

**Parámetros principales:**
- `provider`: Proveedor de IA por defecto
- `commit_interval`: Número de imágenes procesadas antes de guardar en BD
- `log_level`: Nivel de logging (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)

## Uso Programático

### Procesamiento de Imágenes

```python
from modules_captions import DatabaseManager, create_client, FileProcessor

# Configurar componentes
db_manager = DatabaseManager("captions.db")
client = create_client("gemini")
processor = FileProcessor(
    root_directory="./imagenes",
    db_manager=db_manager,
    ai_client=client
)

# Procesar imágenes
results = processor.process_images()
print(f"Procesadas: {results['total_processed']} imágenes")
```

### Operaciones de Base de Datos

```python
from modules_captions.db import DatabaseManager

db = DatabaseManager("captions.db")

# Insertar descripción
db.insert_description(
    document_name="documento_001",
    page_number=1,
    image_filename="imagen_001.png",
    description="Descripción de la imagen"
)

# Consultar descripción existente
if db.description_exists("documento_001", 1, "imagen_001.png"):
    descripcion = db.get_description("documento_001", 1, "imagen_001.png")
    print(descripcion)

# Obtener estadísticas
stats = db.get_statistics()
print(f"Total descripciones: {stats['total_descriptions']}")
```

## Base de Datos

El sistema utiliza SQLite con la siguiente estructura:

```sql
CREATE TABLE image_descriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_name TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    image_filename TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_name, page_number, image_filename)
);
```

## Logging

### Archivos Generados

- `logs/caption_extractor_YYYYMMDD.log`: Registro principal
- `logs/errors_YYYYMMDD.json`: Errores detallados
- `logs/completed_directories.json`: Directorios procesados

### Niveles de Log

| Nivel | Código | Descripción |
|-------|--------|-------------|
| DEBUG | 10 | Información detallada |
| INFO | 20 | Progreso general |
| WARNING | 30 | Advertencias |
| ERROR | 40 | Errores de operación |



## Solución de Problemas

### Errores Comunes

| Error | Causa | Solución |
|-------|-------|----------|
| API Key no encontrada | Variable de entorno no configurada | Configurar `PROVIDER_API_KEY` |
| Permisos de base de datos | Sin permisos de escritura | Verificar permisos del directorio |
| Memoria insuficiente | Imágenes muy grandes | Reducir `commit_interval` |
| Rate limit excedido | Demasiadas solicitudes | Usar `--debug` para ver límites |

### Comandos de Depuración

```bash
# Ver estado detallado
python extract_captions.py --status --debug

# Logging completo
python extract_captions.py --root-dir ./imagenes --log-level DEBUG

# Verificar configuración
python -c "from modules_captions import DatabaseManager; print('OK')"
```