import os
import re
from pathlib import Path

def find_shortest_documents(extractions_folder, num_files=5):
    """
    Find the shortest XML documents in the extractions folder.
    
    Args:
        extractions_folder (str): Path to the folder containing XML files
        num_files (int): Number of shortest files to return
    
    Returns:
        list: List of tuples (file_id, file_size) for the shortest documents
    """
    xml_files = []
    
    # Get all XML files and their sizes
    for filename in os.listdir(extractions_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(extractions_folder, filename)
            file_size = os.path.getsize(file_path)
            
            # Extract the ID (rxxx format) from the filename
            match = re.match(r'(r\d{3})', filename)
            if match:
                file_id = match.group(1)
                xml_files.append((file_id, file_size, filename))
    
    # Sort by file size (ascending) and get the shortest ones
    xml_files.sort(key=lambda x: x[1])
    
    return xml_files[:num_files]

def main():
    # Path to the extractions folder
    extractions_folder = "../data/extractions"
    
    # Convert to absolute path
    script_dir = Path(__file__).parent
    extractions_path = (script_dir / extractions_folder).resolve()
    
    print("Finding the 5 shortest XML documents...")
    print("=" * 50)
    
    try:
        shortest_docs = find_shortest_documents(str(extractions_path), 5)
        
        print(f"{'Rank':<5} {'ID':<6} {'Size (bytes)':<12} {'Filename'}")
        print("-" * 60)
        
        for i, (file_id, file_size, filename) in enumerate(shortest_docs, 1):
            print(f"{i:<5} {file_id:<6} {file_size:<12} {filename}")
        
        print("\nIDs of the 5 shortest documents:")
        ids_only = [doc[0] for doc in shortest_docs]
        print(", ".join(ids_only))
        
    except FileNotFoundError:
        print(f"Error: Could not find the extractions folder at {extractions_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()