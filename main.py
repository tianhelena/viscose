print('hello')  
import spacy
import spacy
from gensim.models import Word2Vec

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Sample transaction description
transaction = "PAYPAL *EBAY 12345678901 CA"

# Apply NER
doc = nlp(transaction)
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Load pre-trained Word2Vec model
model = Word2Vec.load("path_to_pretrained_word2vec.model")
