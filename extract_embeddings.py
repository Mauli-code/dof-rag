# %%
"""
DOF Document Embedding Generator with Structured Headers

This script processes markdown files from the Mexican Official Gazette (DOF),
extracts their content with advanced header structure recognition, splits them 
into page-based chunks maintaining hierarchical context, and generates vector
embeddings for efficient semantic search. The embeddings are stored in a SQLite
database with vector search capabilities.

NEW FEATURES:
- Structured header detection and hierarchical context maintenance
- Page-based chunking using {number}----- patterns
- Contextual headers for improved embedding quality
- Debug output files for manual inspection

Usage:
python extract_embeddings.py /path/to/markdown/files [--verbose]
"""

import os
import re
import logging
from datetime import datetime
from typing import Union, Tuple, Dict

import typer
from fastlite import database
from sqlite_vec import load
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dof_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dof_embeddings")

# %%

# Model configuration
model = SentenceTransformer("nomic-ai/modernbert-embed-base", trust_remote_code=True)

# %%
# Database initialization and schema setup
# Using SQLite with sqlite-vec extension for vector similarity search
db = database("dof_db/db.sqlite")
db.conn.enable_load_extension(True)
load(db.conn)
db.conn.enable_load_extension(False)

# Documents table: stores metadata about each document
db.t.documents.create(
    id=int, title=str, url=str, file_path=str, created_at=datetime, pk="id", ignore=True
)
db.t.documents.create_index(["url"], unique=True, if_not_exists=True)

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

HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.*)$')

def get_heading_level(line: str) -> Union[Tuple[int, str], Tuple[None, None]]:
    """
    Returns the heading level and text,
    or (None, None) if the line is not a heading.
    """
    match = HEADING_PATTERN.match(line)
    if match:
        hashes = match.group(1)
        heading_text = match.group(2).strip()
        level = len(hashes)
        return level, heading_text
    return None, None


def update_open_headings_dict(open_headings: Dict[int, str], line: str) -> Dict[int, str]:
    """
    Updates the dictionary of open headings according to the line.
    
    - If an H1 is found, all previous headings are removed.
    - If a heading of any level is found, all lower priority headings (higher numbers) 
      are removed and the current one is added or updated.
    - If no heading is found, the dictionary remains unchanged.
    
    Args:
        open_headings: Current dictionary of headings (level -> text)
        line: The line to check for headings
        
    Returns:
        Updated dictionary of headings
    """
    lvl, txt = get_heading_level(line)
    if lvl is None:
        # No heading in this line
        return open_headings
    
    if lvl == 1:
        # H1 resets all context
        logger.debug(f"H1 heading found: '{txt}'. Resetting heading context")
        return {1: txt}
    else:
        # Create copy to avoid modifying the original
        new_headings = {k: v for k, v in open_headings.items() if k < lvl}
        new_headings[lvl] = txt
        logger.debug(f"H{lvl} heading found: '{txt}'. Updating context")
        return new_headings


def build_header_dict(doc_title: str, page: str, open_headings: Dict[int, str]) -> str:
    """
    Builds the chunk header with the format:
    
    # Document: <Document Name> | page: <Page Number>
    ## Heading Level 2
    ### Heading Level 3
    ...
    
    The headings are listed in order of level (H1, H2, H3, etc.)
    
    Args:
        doc_title: Title of the document
        page: Page number
        open_headings: Dictionary of open headings (level -> text)
        
    Returns:
        Formatted header as a string
    """
    header_lines = [f"# Document: {doc_title} | page: {page}"]
    
    for level in sorted(open_headings.keys()):
        text = open_headings[level]
        header_lines.append(f"{'#' * level} {text}")
    
    return "\n".join(header_lines)


def split_text_by_page_break(text: str):
    """
    Splits the text into chunks based on the page pattern:
    {number} followed by at least 5 hyphens.
    """
    page_pattern = re.compile(r'\{(\d+)\}\s*-{5,}')
    chunks = []
    last_index = 0
    last_page = None

    for match in page_pattern.finditer(text):
        page_num = match.group(1)
        chunk_text = text[last_index:match.start()].strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "page": page_num})
        last_index = match.end()
        last_page = page_num

    # Last fragment after the last page mark
    remaining = text[last_index:].strip()
    if remaining:
        final_page = last_page if last_page else "1"
        chunks.append({"text": remaining, "page": final_page})

    logger.debug(f"Document split into {len(chunks)} chunks based on page markers")
    return chunks

# %%


def get_url_from_filename(filename: str):
    """
    Generate the URL based on the filename pattern.

    Expected filename format: DDMMYYYY-XXX.md
    Example: 23012025-MAT.md (representing January 23, 2025, MAT section)

    The generated URL points to the PDF document in the DOF repository.

    Args:
        filename (str): Filename of the .md file

    Returns:
        str: The URL to the original PDF document

    Raises:
        ValueError: If the filename doesn't match the expected format
    """
    # Extract just the base filename in case the full path was passed
    base_filename = os.path.basename(filename).replace(".md", "")
    
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


def process_file(file_path, verbose: bool = False):
    """
    Process a markdown file with structured headers support.
    This function:
    1. Reads file content
    2. Extracts metadata from filename
    3. Deletes any previous version of this document
    4. Splits content by page breaks using the improved algorithm
    5. Maintains hierarchical context with open headings
    6. Generates contextual headers for each chunk
    7. Generates vector embeddings for each chunk (including header)
    8. Stores chunks and embeddings in the database
    9. Creates a debug file for manual inspection (only if verbose=True)

    Args:
        file_path (str): Path to the markdown file to process
        verbose (bool): If True, creates debug chunks file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        title = os.path.splitext(os.path.basename(file_path))[0]
        url = get_url_from_filename(file_path)
        logger.debug(f"Processing file '{file_path}' with title '{title}'")

        # Delete any existing document with the same URL to avoid duplicates
        db.t.documents.delete_where("url = ?", [url])
        doc = db.t.documents.insert(
            title=title,
            url=url,
            file_path=file_path,
            created_at=datetime.now()
        )

        # Divide the content into chunks based on page breaks
        if re.search(r'\{\d+\}\s*-{5,}', content):
            page_chunks = split_text_by_page_break(content)
        else:
            page_chunks = [{"text": content, "page": "1"}]
            logger.debug("No page markers found. Processing as a single-page document.")

        open_headings_dict = {}  # Dictionary of (level -> text) that are "open"
        chunk_counter = 0
        chunks_file = None
        chunks_file_path = None
        
        if verbose:
            chunks_file_path = os.path.splitext(file_path)[0] + "_chunks.txt"
            chunks_file = open(chunks_file_path, "w", encoding="utf-8")
        
        try:
            for chunk in page_chunks:
                chunk_counter += 1
                chunk_text = chunk["text"]
                page_number = chunk["page"]

                # For the first chunk, pre-read the lines until the first heading is obtained.
                if chunk_counter == 1:
                    lines = chunk_text.splitlines()
                    for line in lines:
                        lvl, txt = get_heading_level(line)
                        if lvl is not None:
                            # Update headings dictionary directly
                            open_headings_dict = update_open_headings_dict(open_headings_dict, line)
                            # If the first heading is H1, we stop the pre-reading.
                            if lvl == 1:
                                break
                        else:
                            # Stop if the first line that is not a heading is found.
                            break

                # Build the header using the "open" headings at the beginning of the chunk.
                header = build_header_dict(title, page_number, open_headings_dict)

                # Prepare text to generate the embedding.
                text_for_embedding = f"{header}\n\n{chunk_text}"
                
                try:
                    embedding = model.encode(f"search_document: {text_for_embedding}")
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk #{chunk_counter} of {file_path}: {str(e)}")
                    raise

                # Write the chunk details to the _chunks.txt file (only if verbose).
                if verbose and chunks_file:
                    chunks_file.write(f"--- CHUNK #{chunk_counter} ---\n")
                    chunks_file.write(f"Header:\n{header}\n\n")
                    chunks_file.write(f"Text:\n{chunk_text}\n")
                    chunks_file.write("\n" + "-"*50 + "\n\n")

                # Save the chunk in the database.
                db.t.chunks.insert(
                    document_id=doc["id"],
                    text=chunk_text,
                    header=header,
                    embedding=embedding,
                    created_at=datetime.now(),
                )

                # Update the state of open headings for the following chunks.
                lines = chunk_text.splitlines()
                for line in lines:
                    open_headings_dict = update_open_headings_dict(open_headings_dict, line)

        finally:
            if chunks_file:
                chunks_file.close()
        
        logger.info(f"Processing completed for: {file_path}")
        if verbose and chunks_file_path:
            logger.info(f"Chunks file generated at: {chunks_file_path}")
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        raise


def process_directory(directory_path, verbose: bool = False):
    """
    Recursively process all files in a directory and its subdirectories.

    Args:
        directory_path (str): Path to the directory to process
        verbose (bool): If True, creates debug chunks files
    """
    try:
        # Get the list of files in the directory
        entries = os.listdir(directory_path)
        md_files = [entry for entry in entries if entry.lower().endswith(".md")]
        logger.info(f"Found {len(md_files)} markdown files in {directory_path}")
        
        # Loop through all entries in the directory
        for entry in tqdm(entries, desc=f"Processing {directory_path}"):
            # Create full path
            entry_path = os.path.join(directory_path, entry)

            # If it's a file, process it
            if os.path.isfile(entry_path) and entry_path.lower().endswith(".md"):
                process_file(entry_path, verbose=verbose)
            # If it's a directory, recursively process it
            elif os.path.isdir(entry_path):
                process_directory(entry_path, verbose=verbose)
                
        logger.info(f"Processing completed for directory: {directory_path}")
    except Exception as e:
        logger.error(f"Error processing directory {directory_path}: {str(e)}")
        raise


def main(root_dir: str, verbose: bool = False):
    """
    Process all markdown files in a directory and its subdirectories.
    
    Args:
        root_dir (str): Root directory to search for markdown files
        verbose (bool, optional): If True, shows detailed debug messages. Default is False.
    """
    # Configure logging level based on verbose parameter
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode activated - showing detailed debug messages")
    else:
        logger.setLevel(logging.INFO)
        
    logger.info(f"Starting document processing in: {root_dir}")
    start_time = datetime.now()
    
    try:
        process_directory(root_dir, verbose=verbose)
        elapsed_time = datetime.now() - start_time
        logger.info(f"Processing completed in {elapsed_time}")
    except Exception as e:
        logger.error(f"Error in processing: {str(e)}")
        raise


if __name__ == "__main__":
    typer.run(main)
