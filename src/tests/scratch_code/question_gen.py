#%%
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from keybert import KeyBERT
import os
import sys

# Add the path to the parent and grandparent directories to augment search for module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#from component.text_processor import TextProcessor


'''
- scores should be thought of as: document relevance to the document text, and represents the highest scoring key phrase/word
    o x_word has to do with y% of the document

- keybert returns two columns
        o keyphrases: the key phrases extracted from the text (ngram range 2-3)
        o keyphrases_probs: the relevance (1 most relevant) of the key phrase as it relates to the main topic/theme of the input text
        o requirements: tensorflow v2.10.0, transformers v4.44.2, keras v2.10.0, protobuf v3.19.6; uninstall tf-keras
'''

# Function to highlight the extracted key phrase in the context
def highlight_answer(context, answer):
    context_splits = context.split(answer)
    text = f" <h> {answer} <h> ".join(context_splits)
    return text

# Function to prepare the instruction prompt for question generation
def prepare_instruction(answer_highlighted_context):
    # instruction to llm to generate a question based on the highlighted answer
    instruction_prompt = f"""Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks. context: ```{answer_highlighted_context}```"""
    return instruction_prompt

# Function to the key subject using TF-IDF and SVD
def keywords_tfkidf(text, n_components=1):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform([text])
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    terms = vectorizer.get_feature_names_out()
    components = svd.components_[0]
    key_term_index = components.argmax()
    key_phrase = terms[key_term_index]
    score = components[key_term_index]  # Get the score of the key phrase
    return key_phrase, score

def keyphrases_keybert(text, model, top_n=1):
    # setting ngram range to 2, 3 to get two-to-three word phrases instead of single keywords
    keyphrases_with_probs = model.extract_keywords(text, 
                                      keyphrase_ngram_range=(2, 3), 
                                      use_maxsum=True, 
                                      nr_candidates=20, 
                                      top_n=top_n)
    
    # get phrases and scores separately
    keyphrases = [phrase for phrase, _ in keyphrases_with_probs]
    keyprases_probs = [prob for _, prob in keyphrases_with_probs]
    
    return keyphrases, keyprases_probs

# Function to process the DataFrame, extract key phrases, and generate questions
def generate_questions(df):
    # extract keyword+score with tfidf
    df['keyword_tfidf'], df['keyword_tfidf_score'] = zip(*df['clean_text'].apply(lambda text: keywords_tfkidf(text, n_components=1)))
    
    # extract key phrases with keybert
    model = KeyBERT('t5-small') # order from best to lest best: t5-small, t5-base, distilbert-base-nli-mean-tokens
    df['keyphrase_keybert'], df['keyphrase_keybert_score'] = zip(*df['clean_text'].apply(lambda text: keyphrases_keybert(text, model, top_n=1)))
    df['keyphrase_keybert'] = df['keyphrase_keybert'].apply(lambda phrases: phrases[0] if phrases else '')

    # highlight the key text in the context
    df['highlighted_context_keyword'] = df.apply(lambda row: highlight_answer(row['clean_text'], row['keyword_tfidf']), axis=1)
    df['highlighted_context_keyphrase'] = df.apply(lambda row: highlight_answer(row['clean_text'], row['keyphrase_keybert']), axis=1)
    
    # prepare the instruction prompt for question generation
    df['instruction_prompt_keyword'] = df['highlighted_context_keyword'].apply(prepare_instruction)
    df['instruction_prompt_keyphrase'] = df['highlighted_context_keyphrase'].apply(prepare_instruction)
    
    # generate questions based on the keyword and key phrase
    df['keyword_generated_question'] = df['instruction_prompt_keyword'].apply(lambda prompt: pipe(prompt, num_return_sequences=1, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)[0]['generated_text'])
    df['keyphrase_generated_question'] = df['instruction_prompt_keyphrase'].apply(lambda prompt: pipe(prompt, num_return_sequences=1, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)[0]['generated_text'])
      
    return df


# Load the CSV file to a DataFrame
input_csv = 'subset_for_examine100.csv'

if not os.path.exists(input_csv):
    print(f"Error: {input_csv} not found.")
    sys.exit()

df = pd.read_csv(input_csv)
df = df.head(10)

# Initialize the pipeline for question generation
pipe = pipeline('text2text-generation', model='valhalla/t5-small-qg-hl') # mohammedaly22/t5-small-squad-qg

# Generate questions based on the keyword and key phrase
df_with_questions = generate_questions(df)

# Print results
df_with_questions.head()

# Save the DataFrame to a CSV file
output_csv = 'questions_generated.csv'
df_with_questions.to_csv(output_csv, index=False)

# TRY DOING POS FOR KEYWORD EXTRACTION
# TRY USING CODE TO IDENTIFY THE BEST FORMATTED/STRUCTURED QUESTION
