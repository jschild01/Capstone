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

# Correct grammar mistakes in 'keyphrase_generated_question'
df['keyword_generated_question_val'] = df['keyword_generated_question'].apply(has_subject_and_verb)
df['keyphrase_generated_question_val'] = df['keyphrase_generated_question'].apply(has_subject_and_verb)

# Remove instances where both keyword and keyphrase generated questions are not valid
df = df[df['keyword_generated_question_val'] | df['keyphrase_generated_question_val']]
print(len(df))

# Replaces faulty questions without proper subj-verb with 'Faulty question'
df.loc[~df['keyword_generated_question_val'], 'keyword_generated_question'] = 'Faulty question'
df.loc[~df['keyphrase_generated_question_val'], 'keyphrase_generated_question'] = 'Faulty question'

# Save the filtered data
df.to_csv('afc_txtFiles_QA_filtered.csv', index=False)









#%%

# read csv file
df_og = pd.read_csv('afc_txtFiles_QA_filtered.csv')
df = df_og.copy()

df.head(10)







