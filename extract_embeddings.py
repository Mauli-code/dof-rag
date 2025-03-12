# %%

import os
from datetime import datetime

import typer
from fastlite import database
from sqlite_vec import load, serialize_float32
from semantic_text_splitter import MarkdownSplitter
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tqdm import tqdm

# %%

tokenizer = Tokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
model = SentenceTransformer("nomic-ai/modernbert-embed-base", trust_remote_code=True)
splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, 2048)

# %%

db = database("dof_db/db.sqlite")
db.conn.enable_load_extension(True)
load(db.conn)
db.conn.enable_load_extension(False)

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

db.t.chunks.create(
    id=int,
    document_id=int,
    text=str,
    embedding=bytes,
    created_at=datetime,
    pk="id",
    foreign_keys=[("document_id", "documents")],
    ignore=True
)

# %%

def get_url_from_filename(filename: str):
    """
    Generate the URL based on the filename pattern '23012025-MAT.pdf'.

    Args:
        filename (str): Filename of the .md file

    Returns:
        str: The generated URL
    """
    # Extract just the base filename in case the full path was passed
    base_filename = os.path.basename(filename).replace('.md', '')

    # The year should be extracted from the filename (positions 4-8 in 23012025-MAT.pdf)
    # This assumes the format is consistent
    if len(base_filename) >= 8:
        year = base_filename[4:8]  # Extract year (2025 from 23012025-MAT.pdf)
        pdf_filename = f"{base_filename}.pdf"  # Add .pdf extension back

        # Construct the URL
        url = f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_filename}&anio={year}&repo=repositorio/"
        return url
    else:
        # Return None or an error message if the filename doesn't match expected format
        raise ValueError(f"Expected filename like 23012025-MAT.md but got {filename}")

def process_file(file_path):
    """
    Process a file. Replace this function with your specific processing logic.

    Args:
        file_path (str): Path to the file to process
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # Get the title (filename without extension)
        title = os.path.splitext(os.path.basename(file_path))[0]
        url = get_url_from_filename(file_path)

        db.t.documents.delete_where('url = ?', [url])

        doc = db.t.documents.insert(
            title=title,
            url=url,
            file_path=file_path,
            created_at=datetime.now()
        )

        chunks = splitter.chunks(content)

        # Store chunks and embeddings
        for i, chunk in enumerate(tqdm(chunks, desc=f"Processing {file_path}")):
            embedding = model.encode(f"search_document: {chunk}")
            db.t.chunks.insert(
                document_id=doc['id'],
                text=chunk,
                embedding=embedding,
                created_at=datetime.now()
            )


def process_directory(directory_path):
    """
    Recursively process all files in a directory and its subdirectories.

    Args:
        directory_path (str): Path to the directory to process
    """
    # Loop through all entries in the directory
    for entry in tqdm(os.listdir(directory_path), desc=f"Processing {directory_path}"):
        # Create full path
        entry_path = os.path.join(directory_path, entry)

        # If it's a file, process it
        if os.path.isfile(entry_path) and entry_path.lower().endswith('.md'):
            process_file(entry_path)
        # If it's a directory, recursively process it
        elif os.path.isdir(entry_path):
            process_directory(entry_path)


def main(root_dir: str):
    process_directory(root_dir)


if __name__ == '__main__':
    typer.run(main)
