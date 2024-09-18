import os
import pandas as pd
from tkinter import Tk, filedialog

def select_directory():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

def get_mimetype_choice():
    mimetypes = [
        'application/pdf',
        'video/mp4',
        'audio/mp3',
        'text/plain',
        'application/msword',
        'application/epub+zip'
    ]
    
    print("Please select a mimetype:")
    for i, mimetype in enumerate(mimetypes, 1):
        print(f"{i}. {mimetype}")
    
    while True:
        try:
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(mimetypes):
                return mimetypes[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def process_csv_files(root_dir, target_mimetype):
    results = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'file_list.csv':
                file_path = os.path.join(dirpath, filename)
                df = pd.read_csv(file_path)
                
                if 'mimetype' in df.columns and 'source_url' in df.columns:
                    # Filter out rows where either mimetype or source_url is NaN
                    df = df.dropna(subset=['mimetype', 'source_url'])
                    
                    matching_rows = df[df['mimetype'] == target_mimetype]
                    results.extend(matching_rows['source_url'].tolist())
    
    return results

def main():
    print("Select the root directory containing your collection folders:")
    root_dir = select_directory()
    
    target_mimetype = get_mimetype_choice()
    
    print(f"\nSearching for files with mimetype: {target_mimetype}")
    source_urls = process_csv_files(root_dir, target_mimetype)
    
    print(f"\nFound {len(source_urls)} matching files:")
    for url in source_urls:
        print(url)
    
    # Optionally, save results to a file
    save_option = input("\nDo you want to save these URLs to a file? (y/n): ").lower()
    if save_option == 'y':
        output_file = os.path.join(root_dir, f"{target_mimetype.replace('/', '_')}_source_urls.txt")
        with open(output_file, 'w') as f:
            for url in source_urls:
                f.write(f"{url}\n")
        print(f"URLs saved to: {output_file}")

if __name__ == "__main__":
    main()
