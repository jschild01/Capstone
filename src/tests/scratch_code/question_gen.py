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
#from component.rag_text_processor import TextProcessor


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
    #instruction_prompt = f"""Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks. context: ```{answer_highlighted_context}```"""
    #instruction_prompt = (
    #    f"Given the context below, generate a question that is directly and exclusively related to the part highlighted by <h>. "
    #    f"Do not generate questions based on unrelated parts of the context. "
    #    f"Only use the highlighted portion as the answer reference. "
    #    f"Context is provided between triple backticks: ```{answer_highlighted_context}```"
    #)

    instruction_prompt = (
        f"### Instruction ###\n"
        f"Generate a question directly related to the highlighted part marked by <h> in the provided context below.\n"
        f"Ignore all other parts of the context; focus only on the highlighted section as the answer reference.\n\n"
        f"### Context ###\n"
        f"```{answer_highlighted_context}```"
    )
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
                                      keyphrase_ngram_range=(1, 1), 
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
    model = KeyBERT('t5-large') # order from best to lest best: t5-small, t5-base, distilbert-base-nli-mean-tokens
    df['keyphrase_keybert'], df['keyphrase_keybert_score'] = zip(*df['clean_text'].apply(lambda text: keyphrases_keybert(text, model, top_n=1)))
    df['keyphrase_keybert'] = df['keyphrase_keybert'].apply(lambda phrases: phrases[0] if phrases else '')

    # highlight the key text in the context
    df['highlighted_context_keyword'] = df.apply(lambda row: highlight_answer(row['clean_text'], row['keyword_tfidf']), axis=1)
    df['highlighted_context_keyphrase'] = df.apply(lambda row: highlight_answer(row['clean_text'], row['keyphrase_keybert']), axis=1)
    
    # prepare the instruction prompt for question generation
    df['instruction_prompt_keyword'] = df['highlighted_context_keyword'].apply(prepare_instruction)
    df['instruction_prompt_keyphrase'] = df['highlighted_context_keyphrase'].apply(prepare_instruction)
    
    # generate questions based on the keyword and key phrase
    df['keyword_generated_question'] = df['instruction_prompt_keyword'].apply(lambda prompt: pipe(prompt, num_return_sequences=2, num_beams=10, num_beam_groups=10, diversity_penalty=1.0)[0]['generated_text'])
    df['keyphrase_generated_question'] = df['instruction_prompt_keyphrase'].apply(lambda prompt: pipe(prompt, num_return_sequences=2, num_beams=10, num_beam_groups=10, diversity_penalty=1.0)[0]['generated_text'])
      
    return df






# Load the CSV file to a DataFrame
input_csv = 'subset_for_examine100.csv'

if not os.path.exists(input_csv):
    print(f"Error: {input_csv} not found.")
    sys.exit()

df = pd.read_csv(input_csv)
df = df.head(10)

# Initialize the pipeline for question generation
pipe = pipeline('text2text-generation', model='mohammedaly22/t5-small-squad-qg') # valhalla/t5-small-qg-hl

# Generate questions based on the keyword and key phrase
df_with_questions = generate_questions(df)

# Print results
df_with_questions.head()

# Save the DataFrame to a CSV file
output_csv = 'questions_generated.csv'
df_with_questions.to_csv(output_csv, index=False)

# TRY DOING POS FOR KEYWORD EXTRACTION
# TRY USING CODE TO IDENTIFY THE BEST FORMATTED/STRUCTURED QUESTION






















#%%

import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from keybert import KeyBERT
import pandas as pd
import os
import sys

# Combined function to trim context, highlight the answer, and prepare the instruction prompt
def process_context_and_prepare_instruction(context, answer):
    # Split context into sentences
    sentences = re.split(r'(?<=[.!?]) +', context)
    
    # Find the index of the sentence containing the answer
    sentence_index = next((i for i, sentence in enumerate(sentences) if answer in sentence), None)
    
    # Extract the sentence containing the answer and its surrounding sentences
    if sentence_index is not None:
        start = max(0, sentence_index - 2)
        end = min(len(sentences), sentence_index + 3)
        trimmed_context = ' '.join(sentences[start:end])
    else:
        trimmed_context = context  # Fall back to full context if no match found
    
    # Highlight the answer in the trimmed context
    highlighted_context = trimmed_context.replace(answer, f" <h> {answer} <h> ")

    # Prepare the instruction prompt
    instruction_prompt = (
        f"### Instruction ###\n"
        f"Generate a question directly related to the highlighted part marked by <h> in the provided context below.\n"
        f"Ignore all other parts of the context; focus only on the highlighted section as the answer reference.\n\n"
        f"### Context ###\n"
        f"```{highlighted_context}```"
    )
    
    return instruction_prompt

# Function to extract the key subject using TF-IDF and SVD
def keywords_tfidf(text, n_components=1):
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

# Function to extract key phrases using KeyBERT
def keyphrases_keybert(text, model, top_n=1):
    keyphrases_with_probs = model.extract_keywords(
        text, keyphrase_ngram_range=(1, 1), use_maxsum=True, nr_candidates=20, top_n=top_n
    )
    keyphrases = [phrase for phrase, _ in keyphrases_with_probs]
    keyphrases_probs = [prob for _, prob in keyphrases_with_probs]
    return keyphrases, keyphrases_probs

def get_token_count(text):
    return len(text.split())

# Function to process the DataFrame, extract key phrases, and generate questions
def generate_questions(df):
    # get token count for filtering out short documents for which questions may not be well generated
    df['token_count'] = df['clean_text'].apply(get_token_count)

    # Extract keyword and score with TF-IDF
    df['keyword_tfidf'], df['keyword_tfidf_score'] = zip(
        *df['clean_text'].apply(lambda text: keywords_tfidf(text, n_components=1))
    )
    
    # Extract key phrases with KeyBERT
    model = KeyBERT('t5-large')
    df['keyphrase_keybert'], df['keyphrase_keybert_score'] = zip(
        *df['clean_text'].apply(lambda text: keyphrases_keybert(text, model, top_n=1))
    )
    df['keyphrase_keybert'] = df['keyphrase_keybert'].apply(lambda phrases: phrases[0] if phrases else '')
    
    # Prepare the instruction prompt by trimming and highlighting key phrases
    df['instruction_prompt_keyword'] = df.apply(
        lambda row: process_context_and_prepare_instruction(row['clean_text'], row['keyword_tfidf']), axis=1
    )
    df['instruction_prompt_keyphrase'] = df.apply(
        lambda row: process_context_and_prepare_instruction(row['clean_text'], row['keyphrase_keybert']), axis=1
    )
    
    # Generate questions based on the keyword and key phrase
    df['keyword_generated_question'] = df['instruction_prompt_keyword'].apply(
        lambda prompt: pipe(prompt, num_return_sequences=1, num_beams=2, num_beam_groups=2, diversity_penalty=1.0)[0]['generated_text']
    )
    df['keyphrase_generated_question'] = df['instruction_prompt_keyphrase'].apply(
        lambda prompt: pipe(prompt, num_return_sequences=1, num_beams=2, num_beam_groups=2, diversity_penalty=1.0)[0]['generated_text']
    )
    
    return df

# Load the CSV file to a DataFrame
input_csv = 'subset_for_examine100.csv'

if not os.path.exists(input_csv):
    print(f"Error: {input_csv} not found.")
    sys.exit()

df = pd.read_csv(input_csv)
df = df.head(10)

# Initialize the pipeline for question generation
pipe = pipeline('text2text-generation', model='mohammedaly22/t5-small-squad-qg') # valhalla/t5-base-qa-qg-hl

# Generate questions based on the keyword and key phrase
df_with_questions = generate_questions(df)

# Print results
df_with_questions.head()

# Save the DataFrame to a CSV file
output_csv = 'questions_generated.csv'
df_with_questions.to_csv(output_csv, index=False)
