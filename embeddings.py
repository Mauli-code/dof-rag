import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime
    import os

    from fastlite import database
    from sqlite_vec import load
    return database, datetime, load, os


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(database, load):
    db = database("dof_db/db.sqlite")
    db.conn.enable_load_extension(True)
    load(db.conn)
    db.conn.enable_load_extension(False)
    return (db,)


@app.cell
def _():
    from sqlmodel import create_engine
    DATABASE_URL = "sqlite:///dof_db/db.sqlite"
    engine = create_engine(DATABASE_URL)
    return DATABASE_URL, create_engine, engine


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        create table if not exists documents (
            id integer primary key,
            title text,
            url text unique,
            file_path text,
            created_at text
        );
        """,
        engine=engine
    )
    return (documents,)


@app.cell
def _(engine, mo):
    _df = mo.sql(
        f"""
        create table if not exists chunks (
            id integer primary key,
            document_id integer not null references documents(id) on delete cascade on update cascade,
            content text,
            embedding blob
        );
        """,
        engine=engine
    )
    return


@app.cell
def _(os):
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
            url = f"https://diariooficial.gob.mx/abrirPDF.php?archivo={pdf_filename}&anio={year}&repo=repositorio"
            return url
        else:
            # Return None or an error message if the filename doesn't match expected format
            raise ValueError(f"Expected filename like 23012025-MAT.md but got {filename}")

    get_url_from_filename("23012025.pdf")
    return (get_url_from_filename,)


@app.cell
def _(Document, datetime, db, get_url_from_filename, os):
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

            db["document"]
            document = Document(
                title=title,
                url=url,
                file_path=file_path,
                created_at=datetime.datetime.now()
            )

        

    def process_directory(directory_path):
        """
        Recursively process all files in a directory and its subdirectories.
    
        Args:
            directory_path (str): Path to the directory to process
        """
        # Loop through all entries in the directory
        for entry in os.listdir(directory_path):
            # Create full path
            entry_path = os.path.join(directory_path, entry)
        
            # If it's a file, process it
            if os.path.isfile(entry_path) and entry_path.lower().endswith('.md'):
                process_file(entry_path)
            # If it's a directory, recursively process it
            elif os.path.isdir(entry_path):
                process_directory(entry_path)

    process_directory("./dof_markdown/2025/02/19022025-MAT")
    return process_directory, process_file


if __name__ == "__main__":
    app.run()
