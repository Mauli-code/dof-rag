import os
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from extract_markdown import extract_markdown_from_pdf

# 📌 Función para extraer PDFs a Markdown
def extract_pdfs_to_markdown(source_dir: str, target_dir: str):
    """
    Itera sobre los archivos PDF en el directorio de origen, extrae el contenido en Markdown
    y lo guarda en la estructura de carpetas dentro del directorio de destino.

    Args:
        source_dir (str): Directorio raíz donde están los PDFs organizados en subcarpetas.
        target_dir (str): Directorio raíz donde se guardarán los archivos Markdown.
    """
    for month in sorted(os.listdir(source_dir)):
        month_path = os.path.join(source_dir, month)

        if not os.path.isdir(month_path):
            continue  # Ignorar archivos que no sean carpetas

        target_month_path = os.path.join(target_dir, month)
        os.makedirs(target_month_path, exist_ok=True)

        for pdf_file in sorted(os.listdir(month_path)):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(month_path, pdf_file)
                md_file_name = pdf_file.replace(".pdf", ".md")
                md_path = os.path.join(target_month_path, md_file_name)

                markdown_text = extract_markdown_from_pdf(pdf_path)

                with open(md_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown_text)

                print(f"✅ Procesado: {pdf_path} → {md_path}")

# 📌 Función para extraer metadatos de los archivos
def parse_metadata_from_filename(file_name):
    """Extrae la fecha y categoría desde el nombre del archivo."""
    parts = file_name.split('-')
    if len(parts) >= 2:
        date_part = parts[0]
        try:
            day, month, year = date_part[:2], date_part[2:4], date_part[4:]
            date = f"{year}-{month}-{day}"
        except:
            date = "Unknown"
        category = parts[1].split('.')[0] if len(parts) > 1 else "Unknown"
    else:
        date, category = "Unknown", "Unknown"

    return {"date": date, "category": category}



def generate_faiss_embeddings(md_dir: str, faiss_index_path: str, model_name="hkunlp/instructor-xl"):
    """
    Genera embeddings desde archivos Markdown y los almacena en un índice FAISS.
    """

    # Cargar el modelo de embeddings
    model = SentenceTransformer(model_name)

    embeddings = []
    metadata = []

    for root, _, files in os.walk(md_dir):
        for file in sorted(files):
            if file.endswith(".md"):
                md_path = os.path.join(root, file)

                with open(md_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                # Generar embedding
                embedding = model.encode(text).astype("float32")
                embeddings.append(embedding)

                # Guardar metadatos
                metadata.append({"file": file, "path": md_path})

                print(f"✅ Procesado: {md_path} → Embedding generado")

    # Convertir embeddings a array numpy
    embeddings = np.array(embeddings, dtype="float32")

    # Obtener la dimensión de los embeddings
    dimension = embeddings.shape[1]

    if len(embeddings) >= 100:
        print("🔹 Usando FAISS con clustering (IVFFlat)")
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_L2)

        # ⚠️ **IMPORTANTE: Entrenar FAISS antes de agregar datos**
        if not index.is_trained:
            print("⏳ Entrenando FAISS...")
            index.train(embeddings)

        index.add(embeddings)  # ✅ Ahora sí se puede agregar
    else:
        print("⚠️ Pocos datos, usando FAISS sin clustering (FlatL2)")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

    # Guardamos el índice FAISS
    faiss.write_index(index, faiss_index_path)

    # Guardamos metadatos en JSON
    metadata_path = faiss_index_path.replace(".index", ".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Índice FAISS guardado en: {faiss_index_path}")
    print(f"✅ Metadatos guardados en: {metadata_path}")

    return index, metadata

