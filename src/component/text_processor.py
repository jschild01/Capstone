import re
import string
import spacy
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(nltk_stopwords.words('english'))
        self.nlp = spacy.load("en_core_web_sm")
    
    def preprocess(self, text, lowercase=False, filter_invalid_sents=False, remove_punctuation=False, remove_stopwords=False, use_lemma=False):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'([{}])\1+'.format(re.escape(string.punctuation)), r'\1', text)
        text = re.sub(r'-', ' ', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^A-Za-z0-9.!? ]+', '', text)
        text = re.sub(r'transcribed and reviewed by contributors participating in the by the people project at crowd.loc.gov.', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()

        if filter_invalid_sents:
            text = self.filter_valid_sentences(text)

        if lowercase:
            text = text.lower()

        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        words = text.split()
        if remove_stopwords:
            words = [word for word in words if word not in self.stop_words]

        if use_lemma:
            words = [self.lemmatizer.lemmatize(word) for word in words]

        return ' '.join(words)

    def is_valid_sentence(self, sentence):
        doc = self.nlp(sentence)
        has_subject = any(token.dep_ in ("nsubj", "nsubjpass") for token in doc)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        return has_subject and has_verb

    def filter_valid_sentences(self, text):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        valid_sentences = [sent for sent in sentences if self.is_valid_sentence(sent)]
        return " ".join(valid_sentences)
