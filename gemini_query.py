"""
Sistema de Consulta de Documentos del DOF con Gemini 2.0

Este módulo implementa un sistema avanzado de búsqueda semántica y consulta mediante 
Gemini 2.0 para documentos del Diario Oficial de la Federación (DOF).

Características:
- Búsqueda vectorial con distancia euclidiana
- Obtención de contexto expandido (chunks adyacentes)
- Consultas a Gemini 2.0 con instrucciones específicas
- Modo interactivo para consultas continuas

Uso:
python gemini_query.py [--query "consulta"] [--interactive]
"""

import os
import numpy as np
import typer
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
from fastlite import database
from sentence_transformers import SentenceTransformer

# Cargar variables de entorno
load_dotenv()

# Configurar la API de Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY no está configurada en las variables de entorno")

genai.configure(api_key=api_key)

# Configuración del modelo de embeddings
model = SentenceTransformer("nomic-ai/modernbert-embed-base", trust_remote_code=True)

# Inicialización de la base de datos
db = database("dof_db/db.sqlite")


def deserialize_embedding(blob):
    """
    Convierte un BLOB almacenado en la base de datos a un vector NumPy de tipo float32.
    """
    return np.frombuffer(blob, dtype=np.float32)


def get_all_chunks():
    """
    Obtiene todos los chunks almacenados en la base de datos.
    
    Returns:
        list: Lista de diccionarios con la información de cada chunk
    """
    results = db.query(
        """
        SELECT 
            c.id as chunk_id,
            c.text as chunk_text,
            c.document_id as document_id,
            d.title as doc_title,
            d.url as doc_url,
            c.embedding as embedding_blob
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        ORDER BY c.id;
        """
    )
    return list(results)


def find_relevant_chunks(query, all_chunks, top_k=10):
    """
    Encuentra los chunks más relevantes para la consulta.
    
    Args:
        query (str): Consulta del usuario
        all_chunks (list): Lista de todos los chunks
        top_k (int): Número máximo de chunks a devolver
        
    Returns:
        list: Lista ordenada de chunks más relevantes
    """
    # Generar embedding para la consulta
    query_embedding = model.encode(f"search_query: {query}")
    
    # Calcular distancias
    scored_results = []
    for chunk in all_chunks:
        chunk_embedding = deserialize_embedding(chunk["embedding_blob"])
        distance = np.linalg.norm(query_embedding - chunk_embedding)
        scored_results.append({
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["chunk_text"],
            "document_id": chunk["document_id"],
            "doc_title": chunk["doc_title"],
            "doc_url": chunk["doc_url"],
            "distance": distance,
            "similarity": 1.0 / (1.0 + distance)  # Convertir distancia a similitud
        })
    
    # Ordenar por menor distancia (mayor similitud)
    return sorted(scored_results, key=lambda x: x["distance"])[:top_k]


def get_extended_context(best_chunk, all_chunks):
    """
    Obtiene un contexto extendido incluyendo chunks adyacentes.
    
    Args:
        best_chunk (dict): El chunk más relevante
        all_chunks (list): Lista de todos los chunks
        
    Returns:
        str: Contexto extendido con chunks adyacentes
    """
    # Filtrar chunks del mismo documento
    same_doc_chunks = [c for c in all_chunks if c["document_id"] == best_chunk["document_id"]]
    
    # Ordenar por ID para mantener el orden original
    same_doc_chunks = sorted(same_doc_chunks, key=lambda c: c["chunk_id"])
    
    # Encontrar la posición del mejor chunk
    best_index = None
    for i, chunk in enumerate(same_doc_chunks):
        if chunk["chunk_id"] == best_chunk["chunk_id"]:
            best_index = i
            break
    
    # Construir contexto extendido
    context = ""
    
    if best_index is not None:
        # Añadir chunk anterior si existe
        if best_index > 0:
            context += same_doc_chunks[best_index - 1]["chunk_text"] + "\n\n"
            
        # Añadir el mejor chunk
        context += best_chunk["chunk_text"] + "\n\n"
        
        # Añadir chunk siguiente si existe
        if best_index < len(same_doc_chunks) - 1:
            context += same_doc_chunks[best_index + 1]["chunk_text"]
    else:
        # Si no se encuentra (caso raro), usar solo el mejor chunk
        context = best_chunk["chunk_text"]
    
    return context


def query_gemini(query, verbose=True):
    """
    Realiza una consulta completa: búsqueda de chunks relevantes y generación de respuesta con Gemini.
    
    Args:
        query (str): Consulta del usuario
        verbose (bool): Si debe mostrar información detallada
        
    Returns:
        dict: Diccionario con la respuesta y metadatos
    """
    try:
        if verbose:
            print(f"\n🔍 Procesando consulta: '{query}'")
            print("Obteniendo chunks de la base de datos...")
        
        # Obtener todos los chunks
        all_chunks = get_all_chunks()
        
        if not all_chunks:
            return {
                "success": False,
                "error": "No se encontraron documentos en la base de datos."
            }
            
        if verbose:
            print(f"Se obtuvieron {len(all_chunks)} chunks totales.")
            print("Buscando chunks relevantes...")
        
        # Encontrar chunks relevantes
        top_results = find_relevant_chunks(query, all_chunks)
        
        if not top_results:
            return {
                "success": False,
                "error": "No se encontraron chunks relevantes para esta consulta."
            }
            
        # Obtener el mejor resultado
        best_chunk = top_results[0]
        
        if verbose:
            print(f"Se encontraron {len(top_results)} chunks relevantes.")
            print(f"Mejor chunk: ID={best_chunk['chunk_id']}, Distancia={best_chunk['distance']:.4f}")
            print("Obteniendo contexto extendido...")
        
        # Obtener contexto extendido (con chunks adyacentes)
        extended_context = get_extended_context(best_chunk, all_chunks)
        
        if verbose:
            print("Generando prompt para Gemini...")
        
        # Construir prompt para Gemini
        prompt = (
            f"Utilizando el siguiente contexto extraído de un documento del Diario Oficial de la Federación:\n\n"
            f"---\n{extended_context}\n---\n\n"
            f"Instrucciones:\n"
            f"- Si la consulta está relacionada directamente con el contenido, responde basándote en la información proporcionada.\n"
            f"- Si la consulta es un saludo, despedida o una interacción social común, responde de forma cordial y adecuada.\n"
            f"- Si se solicita información que no se encuentra en el contexto, indica de manera clara que no hay información relevante disponible en el documento.\n"
            f"- Asegúrate de interpretar correctamente la solicitud, contestando únicamente lo que se pregunta y evitando agregar información irrelevante.\n\n"
            f"Pregunta:\n{query}"
        )
        
        if verbose:
            print("Enviando consulta a Gemini...")
        
        # Generar respuesta con Gemini
        model_gemini = genai.GenerativeModel('gemini-2.0-flash')
        response = model_gemini.generate_content(prompt)
        
        # Preparar resultado
        result = {
            "success": True,
            "query": query,
            "response": response.text,
            "source": {
                "title": best_chunk["doc_title"],
                "url": best_chunk["doc_url"],
                "distance": best_chunk["distance"],
                "similarity": best_chunk["similarity"]
            },
            "context": extended_context
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error durante el proceso: {str(e)}"
        }


def display_result(result):
    """
    Muestra el resultado de la consulta de forma amigable.
    
    Args:
        result (dict): Resultado de la consulta
    """
    if not result["success"]:
        print(f"\n❌ Error: {result['error']}")
        return
    
    print("\n" + "="*60)
    print(f"RESPUESTA A: '{result['query']}'")
    print("="*60)
    print(result["response"])
    print("\n" + "-"*60)
    print(f"Fuente: {result['source']['title']}")
    print(f"URL: {result['source']['url']}")
    print(f"Relevancia (distancia): {result['source']['distance']:.4f}")
    print(f"Similitud: {result['source']['similarity']:.4f}")
    print("-"*60)
    print("\nCONTEXTO UTILIZADO:")
    print("-"*60)
    print(result["context"])
    print("="*60)


def interactive_mode():
    """
    Inicia un modo interactivo para realizar consultas continuas.
    """
    print("\n" + "="*60)
    print("Sistema de Consulta Interactiva con Gemini 2.0")
    print("Escribe 'salir' para terminar")
    print("="*60 + "\n")
    
    while True:
        query = input("\nConsulta: ").strip()
        
        if query.lower() in ['salir', 'exit', 'quit']:
            print("¡Hasta pronto!")
            break
            
        if not query:
            continue
            
        result = query_gemini(query)
        display_result(result)


def main(
    query: str = typer.Option(None, "--query", "-q", help="Consulta directa sin modo interactivo"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Iniciar modo interactivo")
):
    """
    Sistema de consulta de documentos del DOF con Gemini 2.0.
    """
    if query:
        # Modo consulta directa
        result = query_gemini(query)
        display_result(result)
    elif interactive:
        # Modo interactivo
        interactive_mode()
    else:
        print("Debe especificar una consulta (--query) o usar el modo interactivo (--interactive)")


if __name__ == "__main__":
    typer.run(main) 