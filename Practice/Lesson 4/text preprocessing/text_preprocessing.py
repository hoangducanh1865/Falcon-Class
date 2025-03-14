import pandas as pd
import re
import unicodedata
import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Dictionary for expanding contractions
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "cz": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "gonna": "going to",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "wanna": "want to",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all're": "you all are",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

def remove_special_char(text, special_characters=['~', '@', '#', '$', '%', '^', '&', '*'], numeric=False):
    pattern = '[' + special_characters[0]
    for char in special_characters:
        pattern = pattern + '|' + char
    if (numeric):
        pattern = pattern + '|' + '0-9'
    pattern = pattern + ']'
    filtered_text = re.sub(pattern, r'', text)
    return filtered_text

def remove_accents(text):
    filtered_text = unicodedata.normalize(
        'NFKD', text).encode('ascii', 'ignore').decode('utf8')
    return filtered_text

def expand_contractions(text):
    text = " ".join([CONTRACTION_MAP[word] if word in CONTRACTION_MAP else word for word in text.split()])
    return text

def remove_stopwords_punctuation(text, lang_model, lemmatizing=False, stop_words=False):
    doc_text = lang_model(text)
    if lemmatizing:
        st = " ".join([token.lemma_ for token in doc_text if not(token.is_punct) and (not stop_words or not token.is_stop)])
    else:
        st = " ".join([token.text for token in doc_text if not(token.is_punct) and (not stop_words or not token.is_stop)])
    return st

def preprocess_text(text, nlp, special_characters=['~', '@', '#', '$', '%', '^', '&', '*'], numeric=False, lemmatizing=False, stop_words=False):
    text = remove_special_char(text, special_characters, numeric)
    text = text.lower().strip()
    text = remove_accents(text)
    text = expand_contractions(text)
    filtered_text = remove_stopwords_punctuation(text, nlp, lemmatizing, stop_words)
    return filtered_text

# Example usage
if __name__ == "__main__":
    # Example 1: Basic text with special characters and contractions
    text1 = "\n\n\nHey that's a $$great news!!"
    processed_text1 = preprocess_text(text1, nlp, lemmatizing=True, stop_words=True)
    print("Example 1:")
    print("Original Text:", text1)
    print("Processed Text:", processed_text1)
    print()

    # Example 2: Text with accents and contractions
    text2 = "C'est déjà l'été, and I'm loving it!"
    processed_text2 = preprocess_text(text2, nlp, lemmatizing=True, stop_words=False)
    print("Example 2:")
    print("Original Text:", text2)
    print("Processed Text:", processed_text2)
    print()

    # Example 3: Text with numbers and special characters
    text3 = "This product costs $100 and it's 5* rated!"
    processed_text3 = preprocess_text(text3, nlp, numeric=True, lemmatizing=False, stop_words=False)
    print("Example 3:")
    print("Original Text:", text3)
    print("Processed Text:", processed_text3)
    print()

    # Example 4: Text with stop words and punctuation
    text4 = "The quick brown fox jumps over the lazy dog."
    processed_text4 = preprocess_text(text4, nlp, lemmatizing=True, stop_words=True)
    print("Example 4:")
    print("Original Text:", text4)
    print("Processed Text:", processed_text4)
    print()

    # Example 5: Text with contractions and mixed case
    text5 = "I'll be there at 5 PM, don't forget!"
    processed_text5 = preprocess_text(text5, nlp, lemmatizing=False, stop_words=False)
    print("Example 5:")
    print("Original Text:", text5)
    print("Processed Text:", processed_text5)
    print()

    # Example 6: Text with only special characters and numbers
    text6 = "12345!@#$%^&*()"
    processed_text6 = preprocess_text(text6, nlp, numeric=True, lemmatizing=False, stop_words=False)
    print("Example 6:")
    print("Original Text:", text6)
    print("Processed Text:", processed_text6)
    print()

    # Example 7: Text with multiple spaces and newlines
    text7 = "  This   is  a   test  \n\n  with  multiple   spaces.  "
    processed_text7 = preprocess_text(text7, nlp, lemmatizing=False, stop_words=False)
    print("Example 7:")
    print("Original Text:", text7)
    print("Processed Text:", processed_text7)
    print()

    # Example 8: Text with all features (accents, contractions, special characters, stop words, etc.)
    text8 = "I'm feeling très bien today! It's 100% amazing, isn't it?"
    processed_text8 = preprocess_text(text8, nlp, lemmatizing=True, stop_words=True)
    print("Example 8:")
    print("Original Text:", text8)
    print("Processed Text:", processed_text8)