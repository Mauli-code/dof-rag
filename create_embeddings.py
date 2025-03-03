import os
from pathlib import Path
from extract_markdown import extract_markdown_from_pdf
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def extract_pdfs_to_markdown(source_dir: str, target_dir: str):
    """
    Itera sobre los archivos PDF en el directorio de origen, extrae el contenido en Markdown
    y lo guarda en la estructura de carpetas dentro del directorio de destino.

    Args:
        source_dir (str): Directorio raíz donde están los PDFs organizados en subcarpetas.
        target_dir (str): Directorio raíz donde se guardarán los archivos Markdown.
    """
    # Iterar sobre los subdirectorios (meses) dentro del directorio de origen
    for month in sorted(os.listdir(source_dir)):
        month_path = os.path.join(source_dir, month)

        # Verificar si es una carpeta
        if not os.path.isdir(month_path):
            continue

        # Crear la carpeta de destino si no existe
        target_month_path = os.path.join(target_dir, month)
        os.makedirs(target_month_path, exist_ok=True)

        # Iterar sobre los archivos PDF dentro del mes
        for pdf_file in sorted(os.listdir(month_path)):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(month_path, pdf_file)
                md_file_name = pdf_file.replace(".pdf", ".md")
                md_path = os.path.join(target_month_path, md_file_name)

                # Extraer el texto en Markdown
                markdown_text = extract_markdown_from_pdf(pdf_path)

                # Guardar el Markdown
                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown_text)

                print(f"Procesado: {pdf_path} → {md_path}")




def generate_faiss_embeddings(md_dir: str, faiss_index_path: str, model_name="hkunlp/instructor-xl"):
    """
    Lee archivos Markdown de un directorio, genera embeddings con el modelo de embeddings 
    'hkunlp/instructor-xl' y los almacena en un índice FAISS.

    Args:
        md_dir (str): Directorio donde están los archivos Markdown.
        faiss_index_path (str): Ruta donde se guardará el índice FAISS.
        model_name (str): Nombre del modelo de embeddings (por defecto 'hkunlp/instructor-xl').
    """
    
    # Cargar el modelo de embeddings
    model = SentenceTransformer(model_name)
    
    # Lista para almacenar los embeddings y textos asociados
    embeddings = []
    metadata = []

    # Iterar sobre los archivos Markdown en el directorio
    for root, _, files in os.walk(md_dir):
        for file in sorted(files):
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                
                # Leer contenido del archivo Markdown
                with open(md_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Generar embedding del texto
                embedding = model.encode(text)
                embeddings.append(embedding)
                
                # Guardar metadatos (nombre del archivo y ruta)
                metadata.append({"file": file, "path": md_path})

                print(f"Procesado: {md_path} → Embedding generado")

    # Convertir lista de embeddings a un array numpy (FAISS requiere esto)
    embeddings = np.array(embeddings, dtype="float32")

    # Crear índice FAISS (uso de L2 como métrica)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    # Agregar los embeddings al índice FAISS
    index.add(embeddings)

    # Guardar el índice en un archivo
    faiss.write_index(index, faiss_index_path)

    print(f"Índice FAISS guardado en: {faiss_index_path}")

    return index, metadata


