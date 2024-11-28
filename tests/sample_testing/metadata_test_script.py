import os
import sys

# Add the src directory to the Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from component.metadata_processor import process_metadata

# Define the path to your real data directory
REAL_DATA_DIR = os.path.join(project_root, 'data', 'marc-xl-data')


def test_process_metadata_with_real_data():
    result = process_metadata(REAL_DATA_DIR)

    # Check if we got any results
    if len(result) == 0:
        print("Error: No metadata was processed")
        return False

    # Check a few expected files or IDs
    if not any('sr01a' in filename for filename in result.keys()):
        print("Error: Expected 'sr01a' file not found in results")
        return False
    if not any('sr01b' in filename for filename in result.keys()):
        print("Error: Expected 'sr01b' file not found in results")
        return False

    # Check for presence of metadata from different sources
    sample_file = next(iter(result))
    sample_metadata = result[sample_file]

    # Updated checks to ensure 'call_number' is a string and not 'N/A'
    if 'call_number' not in sample_metadata or sample_metadata['call_number'] == 'N/A':
        print("Error: Call number not found or invalid in metadata")
        return False
    if 'title' not in sample_metadata:
        print("Error: Title not found in metadata")
        return False
    if 'collection_title' not in sample_metadata:
        print("Error: EAD metadata (collection_title) not found")
        return False
    if 'catalog_title' not in sample_metadata:
        print("Warning: MARC metadata (catalog_title) not found in sample metadata")

    print(f"\nProcessed metadata for {len(result)} files.")
    print(f"\nSample metadata for file '{sample_file}':")
    for key, value in sample_metadata.items():
        print(f"{key}: {value}")

    return True

def test_specific_file_metadata():
    result = process_metadata(REAL_DATA_DIR)

    print(f"Processed files: {list(result.keys())}")

    if 'sr01a_en.txt' not in result:
        print("Error: sr01a_en.txt not found in processed metadata")
        print("Available files:", list(result.keys()))
        return False

    sr01a_metadata = result['sr01a_en.txt']

    print("Successfully found metadata for sr01a_en.txt")

    if sr01a_metadata.get('call_number') == "AFC 1937/002":
        print("Success: Correct call number found for sr01a_en.txt")
    else:
        print(f"Error: Incorrect call number for sr01a_en.txt. Expected 'AFC 1937/002', got '{sr01a_metadata.get('call_number')}'")

    if "House Carpenter" in sr01a_metadata.get('title', ''):
        print("Success: Expected title found for sr01a_en.txt")
    else:
        print("Error: Expected title not found for sr01a_en.txt")

    if 'catalog_title' in sr01a_metadata:
        print("Success: MARC metadata found for sr01a_en.txt")
    else:
        print("Warning: MARC metadata not found for sr01a_en.txt")

    if 'collection_title' in sr01a_metadata:
        print("Success: EAD metadata found for sr01a_en.txt")
    else:
        print("Error: EAD metadata not found for sr01a_en.txt")

    print("\nMetadata for sr01a_en.txt:")
    for key, value in sr01a_metadata.items():
        print(f"{key}: {value}")

    return True

if __name__ == "__main__":
    print("Testing process_metadata_with_real_data:")
    if test_process_metadata_with_real_data():
        print("Test passed successfully.")
    else:
        print("Test failed.")

    print("\nTesting specific_file_metadata:")
    if test_specific_file_metadata():
        print("Test passed successfully.")
    else:
        print("Test failed.")