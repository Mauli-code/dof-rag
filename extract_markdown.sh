#!/bin/bash

# Check if both arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Assign the arguments to variables
INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each PDF file in the input directory
for pdf_file in "$INPUT_DIR"/*.pdf; do
    # Check if there are any PDF files
    if [ ! -f "$pdf_file" ]; then
        echo "No PDF files found in '$INPUT_DIR'."
        exit 1
    fi

    # Get the filename without path
    filename=$(basename "$pdf_file")

    echo "Processing: $filename"

    # Run the marker_single command on each PDF file
    marker_single --output_dir "$OUTPUT_DIR" --paginate_output --languages="es" "$pdf_file"

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed: $filename"
    else
        echo "Error processing: $filename"
    fi
done

echo "All PDF files have been processed."
