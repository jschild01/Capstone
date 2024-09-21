
# Question Generator Script

## Overview

This script is designed to generate questions from a given text dataset, leveraging a combination of text processing, keyword extraction, and question generation models. It uses a pre-trained model (`mohammedaly22/t5-small-squad-qg`) from the Hugging Face Transformers library to create questions based on the context and specific highlighted keywords or phrases.

## Features

- **Data Loading**: Loads a CSV file containing text data for processing.
- **Keyword Extraction**:
  - **TF-IDF Method**: Extracts key terms based on their importance using Term Frequency-Inverse Document Frequency (TF-IDF) combined with Singular Value Decomposition (SVD).
  - **KeyBERT Method**: Uses the KeyBERT model to extract the most relevant keyphrases from the text.
- **Question Generation**: Creates context-specific questions by identifying and focusing on key terms or phrases within the text.
- **Data Saving**: Saves the processed data and generated questions to an output CSV file.

## Dependencies

The script relies on several Python libraries. Install the dependencies using pip:

```bash
pip install pandas transformers scikit-learn keybert
```

## How the Script Works

1. **Initialization**: The `QuestionGenerator` class is initialized with paths to the input and output CSV files and a question generation model.
   
2. **Data Loading**: The script reads the input CSV file, which should contain a column `clean_text` with text data. It processes the first 10 rows of the data.

3. **Keyword and Keyphrase Extraction**:
   - **TF-IDF Extraction**: Applies TF-IDF and SVD to extract the most significant keyword from each text entry.
   - **KeyBERT Extraction**: Extracts keyphrases using the KeyBERT model, ranking the terms by their relevance.

4. **Question Generation**:
   - For each text entry, the script identifies a relevant sentence around the keyword or keyphrase and prepares an instruction prompt.
   - The prompt highlights the keyword or keyphrase, instructing the model to generate a question focusing on the highlighted section.
   - Two questions are generated for each entry: one based on the TF-IDF keyword and one based on the KeyBERT keyphrase.

5. **Saving Results**: The processed data, including generated questions, are saved to the specified output CSV file.

## Script Components

- **`process_context_and_prepare_instruction`**: Prepares the instruction prompt for the question generation model by highlighting the key term or phrase within the context.
- **`keywords_tfidf`**: Extracts the most relevant keyword from the text using TF-IDF and SVD.
- **`keyphrases_keybert`**: Extracts keyphrases from the text using the KeyBERT model.
- **`generate_questions`**: Generates questions using the specified model, based on the prepared prompts.
- **`save_results`**: Saves the generated questions and associated data to the output CSV.

## Usage

To run the script, modify the paths for the input and output CSV files in the main section. The input file should contain a column named `clean_text` with the text data to be processed.

Example usage:

```python
if __name__ == "__main__":
    # Define input and output CSV files
    input_csv = os.path.join(parent, 'data', 'subset_for_examine100.csv')
    output_csv = os.path.join(parent, 'data', 'subset_for_examine100_QA.csv')

    # Instantiate and run the QuestionGenerator
    question_generator = QuestionGenerator(input_csv, output_csv)
    question_generator.run()
```

## Notes

- Ensure that the input CSV file is correctly formatted and located in the specified path.
- The script currently processes the first 10 rows of the data for demonstration purposes. Adjust the `.head(10)` line in the `load_data` method to change this behavior.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
