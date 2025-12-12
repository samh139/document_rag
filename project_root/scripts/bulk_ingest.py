import sys
import os
# Import Pathlib for robust path joining if preferred, but os.path is fine
from app.ingestion.services.chunks_store_service import Job, ingest_file

# Define the directory path from the command line argument
directory_path = sys.argv[1]
owner = sys.argv[2] if len(sys.argv) > 2 else "sameer@example.com"

print(f"Starting ingestion process for all files in: {directory_path}")

# Iterate over all entries in the directory
# os.listdir gets all files and subdirectories
for filename in os.listdir(directory_path):
    # Construct the full file path
    file_path = os.path.join(directory_path, filename)
    
    # Check if the path points to a file and not a directory itself
    if os.path.isfile(file_path) and not filename.startswith('.'):
        print(f"\n--- Processing new file: {filename} ---")
        
        # Create a job for the single file
        job = Job(file_path=file_path, owner=owner)
        
        # Ingest the file
        ingest_file(job)
        
        print(f"Published chunks from {filename}")

print("\nIngestion process complete for all files in the directory.")
