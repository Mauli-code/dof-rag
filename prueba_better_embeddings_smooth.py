"""
DOF Document Embedding Generator

Este script procesa archivos markdown del Diario Oficial de la Federación (DOF),
extrae su contenido, detecta encabezados y subtítulos para crear secciones jerárquicas,
divide cada sección en chunks suaves (split_text_smooth), genera embeddings y los almacena
en una base de datos SQLite con capacidades de búsqueda vectorial.

Basado en:
- Regex para detectar headings en Markdown
- split_text_smooth para dividir texto en oraciones sin romper el flujo
- build_chunk_header para construir encabezados contextuales
"""

import os
import re
from datetime import datetime

import typer
from fastlite import database
from sqlite_vec import load, serialize_float32
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tqdm import tqdm
import google.generativeai as genai
from os import getenv
import numpy as np
from dotenv import load_dotenv

# -----------------------------------------------------
# Configuración de modelos y tokenizadores
# -----------------------------------------------------
tokenizer = Tokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
model = SentenceTransformer("nomic-ai/modernbert-embed-base", trust_remote_code=True)

# Configuración: máximo de caracteres para cada chunk suave
MAX_CHUNK_LENGTH = 1000

# -----------------------------------------------------
# Inicializar la base de datos con sqlite-vec
# -----------------------------------------------------
db = database("dof_db/db.sqlite")
db.conn.enable_load_extension(True)
load(db.conn)
db.conn.enable_load_extension(False)

# -----------------------------------------------------
# Crear/actualizar esquema de tablas
# -----------------------------------------------------
db.t.documents.create(
    id=int, 
    title=str, 
    url=str, 
    file_path=str, 
    created_at=datetime, 
    pk="id", 
    ignore=True
)
db.t.documents.create_index(["url"], unique=True, if_not_exists=True)

# Se añade la columna 'header' para guardar el encabezado contextual
db.t.chunks.create(
    id=int,
    document_id=int,
    text=str,
    header=str,
    embedding=bytes,
    created_at=datetime,
    pk="id",
    foreign_keys=[("document_id", "documents")],
    ignore=True,
)

# -----------------------------------------------------
# Funciones para parsear el documento y dividir el texto suavemente
# -----------------------------------------------------
# Regex para detectar headings en Markdown
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')

def parse_text_by_headings(text: str):
    """
    Divide el texto en secciones basadas en los headings de Markdown.
    Retorna una lista de diccionarios con:
      - "heading_hierarchy": lista con la jerarquía de headings
      - "content": contenido de la sección
    """
    lines = text.split('\n')
    sections = []
    current_hierarchy = []
    current_content_lines = []

    def add_section(hierarchy, content_lines):
        if not hierarchy or not content_lines:
            return
        sections.append({
            "heading_hierarchy": hierarchy.copy(),
            "content": "\n".join(content_lines).strip()
        })

    for line in lines:
        heading_match = HEADING_PATTERN.match(line)
        if heading_match:
            # Cuando se detecta un heading, se cierra la sección anterior
            add_section(current_hierarchy, current_content_lines)
            current_content_lines = []
            hashes = heading_match.group(1)
            heading_text = heading_match.group(2).strip()
            level = len(hashes)
            # Ajustar la jerarquía: mantener los niveles anteriores hasta (nivel-1)
            current_hierarchy = current_hierarchy[:level-1]
            current_hierarchy.append(heading_text)
        else:
            current_content_lines.append(line)

    # Agregar la última sección si existe contenido
    add_section(current_hierarchy, current_content_lines)
    return sections

def split_text_smooth(text: str, max_length: int = MAX_CHUNK_LENGTH, min_chunk_ratio: float = 0.5):
    """
    Divide el texto en chunks suaves sin romper oraciones.
    Separa el texto en oraciones basándose en signos de puntuación y las agrupa sin
    exceder el límite de caracteres (max_length). Si el último chunk es muy corto
    (menos de min_chunk_ratio * max_length), se fusiona con el chunk anterior.
    """
    # Separa el texto en oraciones (manteniendo el signo de puntuación)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        current_chunk = current_chunk.strip()
        # Si el último chunk es muy corto y existe un chunk previo, fusiónalo
        if chunks and len(current_chunk) < max_length * min_chunk_ratio:
            previous_chunk = chunks.pop()
            merged = previous_chunk + " " + current_chunk
            chunks.append(merged.strip())
        else:
            chunks.append(current_chunk)
    return chunks

def build_chunk_header(doc_title: str, heading_hierarchy: list):
    """
    Construye el encabezado contextual utilizando el título del documento y la jerarquía de headings.
    """
    if not heading_hierarchy:
        return f"Document: {doc_title}"
    hierarchy_str = " > ".join(heading_hierarchy)
    return f"Document: {doc_title} | Section: {hierarchy_str}"

# -----------------------------------------------------
# Función para construir la URL del DOF a partir del nombre de archivo
# -----------------------------------------------------
def get_url_from_filename(filename: str) -> str:
    """
    Genera la URL con base en el patrón del nombre de archivo.
    Formato esperado: DDMMYYYY-XXX.md
    Ejemplo: 23012025-MAT.md
    """
    base_filename = os.path.basename(filename).replace(".md", "")
    if len(base_filename) >= 8:
        year = base_filename[4:8]
        pdf_filename = f"{base_filename}.pdf"
        url = f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_filename}&anio={year}&repo=repositorio/"
        return url
    else:
        raise ValueError(f"Expected filename like 23012025-MAT.md but got {filename}")

# -----------------------------------------------------
# Función principal para procesar un archivo y generar embeddings
# -----------------------------------------------------
def process_file(file_path: str):
    """
    Procesa un archivo markdown, extrae secciones basadas en headings,
    divide cada sección en chunks suaves, genera embeddings y los almacena en la base de datos.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # 1) Extraer metadatos y preparar la inserción del documento
    title = os.path.splitext(os.path.basename(file_path))[0]
    url = get_url_from_filename(file_path)

    # Eliminar cualquier versión anterior del documento
    db.t.documents.delete_where("url = ?", [url])
    doc = db.t.documents.insert(
        title=title, 
        url=url, 
        file_path=file_path, 
        created_at=datetime.now()
    )

    # 2) Parsear el contenido basado en headings
    sections = parse_text_by_headings(content)

    # Archivo de salida para depuración de chunks
    chunks_file_path = os.path.splitext(file_path)[0] + "_chunks.txt"
    # Definimos un contador global para numerar secuencialmente los chunks
    global_chunk_counter = 1
    with open(chunks_file_path, "w", encoding="utf-8") as chunks_file:
        # 3) Procesar cada sección
        for section in sections:
            hierarchy = section.get("heading_hierarchy", [])
            section_content = section.get("content", "")
            # Construir el encabezado contextual
            header = build_chunk_header(title, hierarchy)
            # Dividir la sección en chunks suaves
            sub_chunks = split_text_smooth(section_content, max_length=MAX_CHUNK_LENGTH)

            for chunk in sub_chunks:
                # Texto completo para el embedding: encabezado + chunk
                text_for_embedding = f"{header}\n\n{chunk}"
                embedding = model.encode(f"search_document: {text_for_embedding}")

                # Guardar en el archivo de depuración con numeración secuencial
                chunks_file.write(f"--- CHUNK #{global_chunk_counter} ---\n")
                chunks_file.write(f"Header: {header}\n")
                chunks_file.write(f"Texto:\n{chunk}\n")
                chunks_file.write("\n" + "-"*50 + "\n\n")
                global_chunk_counter += 1

                # Insertar en la base de datos
                db.t.chunks.insert(
                    document_id=doc["id"],
                    text=chunk,
                    header=header,
                    embedding=embedding,
                    created_at=datetime.now(),
                )

    print(f"Procesado completado para: {file_path}")
    print(f"Se generó el archivo de chunks en: {chunks_file_path}")

# -----------------------------------------------------
# Función principal para procesar todos los archivos .md de un directorio
# -----------------------------------------------------
def main(directory: str):
    """
    Procesa todos los archivos .md de un directorio.
    """
    md_files = [f for f in os.listdir(directory) if f.endswith(".md")]
    for f in md_files:
        file_path = os.path.join(directory, f)
        process_file(file_path)

if __name__ == "__main__":
    typer.run(main)
