# Generador de Embeddings para Documentos del DOF

## Descripción

Este sistema procesa archivos markdown del Diario Oficial de la Federación (DOF), genera embeddings semánticos y permite realizar consultas utilizando búsqueda vectorial e inteligencia artificial. El sistema divide los documentos en secciones jerárquicas basadas en sus encabezados, fragmenta el texto de manera inteligente y utiliza modelos de embedding avanzados para indexar el contenido, ofreciendo una recuperación precisa de información.

## Características Principales

- **Procesamiento Jerárquico**: Detecta automáticamente encabezados y subtítulos para mantener la estructura del documento.
- **Fragmentación Suave**: Divide el texto respetando límites naturales como oraciones.
- **Generación de Embeddings**: Utiliza modelos modernos para convertir texto a vectores semánticos.
- **Almacenamiento Vectorial**: Base de datos SQLite optimizada para búsquedas vectoriales.
- **Consultas con IA**: Integración con Gemini 2.0 para respuestas contextualizadas.
- **Interfaz Interactiva**: Widgets interactivos para procesamiento y consultas.

## Requisitos

- Python 3.8+
- Librerías: sentence-transformers, fastlite, sqlite-vec, tokenizers, google-generativeai
- API Key de Google para Gemini 2.0

## Instalación

1. Clona el repositorio
2. Instala las dependencias:
3. Configura tu archivo `.env` con tu API Key de Google:


## Estructura de Directorios
├── DOF_Embeddings_Generator.ipynb # Notebook principal
├── dof_db/ # Directorio para la base de datos
│ └── db.sqlite # Base de datos SQLite
├── .env # Archivo de variables de entorno
└── documentos_dof/ # Documentos markdown para procesar


## Uso del Sistema

### Procesamiento de Documentos

1. Ejecuta el notebook `DOF_Embeddings_Generator.ipynb`
2. Usa el widget "Procesamiento de archivos DOF" para ingresar la ruta al directorio con archivos markdown
3. Haz clic en "Procesar archivos" para iniciar la extracción y generación de embeddings

### Visualización de Chunks

1. Utiliza el "Visor de Chunks Generados" para examinar cómo se han dividido los documentos
2. Haz clic en "Buscar archivos" para actualizar la lista de archivos de chunks
3. Selecciona un archivo y presiona "Ver contenido" para visualizar su estructura

### Consultas con Gemini 2.0

1. En la sección "Consulta de Documentos DOF con Gemini 2.0", ingresa tu pregunta
2. Presiona "Consultar" para buscar información relevante
3. El sistema encontrará los fragmentos más similares y generará una respuesta contextualizada

## Proceso Técnico

1. **Extracción de Contenido**: El sistema lee archivos markdown y detecta su estructura jerárquica.
2. **Fragmentación**: Cada sección se divide en chunks de tamaño óptimo (1000 caracteres por defecto).
3. **Generación de Encabezados Contextuales**: Se crea un encabezado para cada fragmento que mantiene el contexto jerárquico.
4. **Embedding**: Se utiliza el modelo "nomic-ai/modernbert-embed-base" para generar vectores semánticos.
5. **Almacenamiento**: Los embeddings y metadatos se guardan en una base de datos SQLite optimizada.
6. **Recuperación Vectorial**: Para consultas, se calcula la similitud entre el embedding de la consulta y los almacenados.
7. **Respuesta con IA**: Se envía el contexto relevante a Gemini 2.0 para generar respuestas precisas.

## Notas Importantes

- El sistema está optimizado para documentos del DOF con formato markdown.
- Los nombres de archivo deben seguir el formato `DDMMYYYY-XXX.md` para generar URLs correctas.
- Para obtener mejores resultados, asegúrate de que los documentos tengan una estructura clara con encabezados markdown (# Título, ## Subtítulo, etc.).
- La calidad de las respuestas depende de la riqueza del contenido indexado.
