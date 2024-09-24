#%%
import pandas as pd
import numpy as np
import spacy

def contains_prohibited_phrases(question, prohibited_phrases):
    return any(phrase in question.lower() for phrase in prohibited_phrases)

def has_subject_and_verb(sentence):
    doc = nlp(sentence)
    has_subject = False
    has_verb = False
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass'):
            has_subject = True
        if token.pos_ == 'VERB':
            has_verb = True
    return has_subject and has_verb

def validate_keyword(keyword): # return true if valid word
    doc = nlp(keyword)
    
    # Filter out words that are less than 4 characters
    if len(keyword) < 4:
        return False
    
    # Check if the part of speech is noun, proper noun, verb, or adjective
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
            return True
    
    return False

# Load the English model in spaCy
nlp = spacy.load("en_core_web_sm")  # Make sure to install this model with 'python -m spacy download en_core_web_sm'

# read csv file
df_og = pd.read_csv('afc_txtFiles_QA.csv')
df = df_og.copy()
print(len(df))

# Keep only rows where 'keyphrase_keybert' and 'keyword_tfidf' is not numeric
df = df[~df['keyphrase_keybert'].str.isnumeric()]
df = df[~df['keyword_tfidf'].str.isnumeric()]
print(len(df))

# Convert the list in 'keyphrase_keybert_score' to float
df['keyphrase_keybert_score'] = df['keyphrase_keybert_score'].str[1:-1]

# Remove any row in which 'keyword_generated_question' doesn't end with a question mark
df = df[df['keyword_generated_question'].str.endswith('?')]
df = df[df['keyphrase_generated_question'].str.endswith('?')]
print(len(df))

# Replaces faulty questions without proper subj-verb with 'Faulty question'
df['keyword_generated_question_val'] = df['keyword_generated_question'].apply(has_subject_and_verb)
df['keyphrase_generated_question_val'] = df['keyphrase_generated_question'].apply(has_subject_and_verb)
df.loc[~df['keyword_generated_question_val'], 'keyword_generated_question'] = 'Faulty question'
df.loc[~df['keyphrase_generated_question_val'], 'keyphrase_generated_question'] = 'Faulty question'

# Remove instances where both keyword and keyphrase generated questions are not valid
df = df[df['keyword_generated_question_val'] | df['keyphrase_generated_question_val']]
print(len(df))

# Replaces faulty keywords not a (subj, adj, verb, pronoun) with 'Faulty keyword'
df['keyword_tfidf_val'] = df['keyword_tfidf'].apply(validate_keyword)
df['keyphrase_keybert_val'] = df['keyphrase_keybert'].apply(validate_keyword)
df.loc[~df['keyword_tfidf_val'], 'keyword_tfidf'] = 'Faulty keyword'
df.loc[~df['keyphrase_keybert_val'], 'keyphrase_keybert'] = 'Faulty keyword'

# Remove instances where both keyword and keyphrase are not valid
df = df[df['keyword_tfidf_val'] | df['keyphrase_keybert_val']]
print(len(df))

# Remove keyword_tfidf_val and keyphrase_keybert_val columns
df = df.drop(columns=['keyword_generated_question_val', 'keyphrase_generated_question_val', 'keyword_tfidf_val', 'keyphrase_keybert_val'])

# extract, clean, organize keyword/tfidf to split up tfidf and keybert derived questions and answers
df_keyword = df[(df['keyword_generated_question'] != 'Faulty question') & (df['keyword_tfidf'] != 'Faulty keyword')]
df_keyword = df_keyword.drop(columns=['token_count', 'keyphrase_keybert', 'keyphrase_keybert_score', 
                                      'instruction_prompt_keyphrase', 'keyphrase_generated_question'])

df_keyword['source_of_keyword'] = 'tfidf'

df_keyword = df_keyword.rename(columns={'keyword_tfidf': 'answer'})
df_keyword = df_keyword.rename(columns={'keyword_tfidf_score': 'relevance_score'})
df_keyword = df_keyword.rename(columns={'instruction_prompt_keyword': 'instruction_prompt'})
df_keyword = df_keyword.rename(columns={'keyword_generated_question': 'generated_question'})


# extract, clean, organize keyphrase/keybert to split up tfidf and keybert derived questions and answers
df_keyphrase = df[(df['keyphrase_generated_question'] != 'Faulty question') & (df['keyphrase_keybert'] != 'Faulty keyword')]
df_keyphrase = df_keyphrase.drop(columns=['token_count', 'keyword_tfidf', 'keyword_tfidf_score', 
                                      'instruction_prompt_keyword', 'keyword_generated_question'])

df_keyphrase['source_of_keyword'] = 'keybert'

df_keyphrase = df_keyphrase.rename(columns={'keyphrase_keybert': 'answer'})
df_keyphrase = df_keyphrase.rename(columns={'keyphrase_keybert_score': 'relevance_score'})
df_keyphrase = df_keyphrase.rename(columns={'instruction_prompt_keyphrase': 'instruction_prompt'})
df_keyphrase = df_keyphrase.rename(columns={'keyphrase_generated_question': 'generated_question'})

# combine the two dataframes
df = pd.concat([df_keyword, df_keyphrase])
print(f'Combined df length: {len(df)}')


# filter to the samples in which the relevance score is greater than 0.4
df = df[df['relevance_score'] > 0.4]
print(len(df))

# Save the filtered data
df.to_csv('afc_txtFiles_QA_filtered.csv', index=False)



