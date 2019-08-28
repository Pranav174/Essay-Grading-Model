import re
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy;
nlp=spacy.load('en')

# Initializing the model and tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('EssayModelRNN.h5')
graph = tf.get_default_graph()

# pre-process the essay. Mainly categorizing common NER
def convert2tokens(essay):
    essay = re.sub("n't"," not",essay)
    essay = re.sub("'re"," are",essay)
    essay = re.sub("it's","it is",essay)
    essay = re.sub("'ve'"," have",essay)
    essay = nlp(essay)
    final=[]
    for token in essay:
        if not token.is_punct:
            if token.ent_iob_=='O':
                final.append(token.text.lower())
            elif token.ent_iob_=='B':
                includeI = False
                if token.ent_type_=='PERSON':
                    final.append('@person')
                elif token.ent_type_=='DATE':
                    final.append('@date')
                elif token.ent_type_=='ORG':
                    final.append('@organization')
                elif token.ent_type_=='TIME':
                    final.append('@time')
                elif token.ent_type_=='MONEY':
                    final.append('@money')
                elif token.ent_type_=='ORDINAL' or token.ent_type_=='CARDINAL':
                    final.append('@num')
                elif token.ent_type_=='PERCENT':
                    final.append('@percent')
                elif token.ent_type_=='GPE' or token.ent_type_=="LOC":
                    final.append('@location')
                else:
                    final.append(token.text.lower())
                    includeI = True
            elif token.ent_iob_=='I' and includeI:
                final.append(token.text.lower())
    return final

# input a list of essays -> outputs list of corresponding marks
def getmarks(essays):
    marks=0
    essays = [convert2tokens(essay) for essay in essays]
    essays = tokenizer.texts_to_sequences(essays)
    MaxSequence = 1200
    essays = pad_sequences(essays, maxlen=MaxSequence)
    with graph.as_default():
        marks = model.predict(essays)
    marks = (marks-5)*1.3+5
    marks = [x[0] for x in marks]
    marks = [min(x,10) for x in marks]
    marks = [max(x,0) for x in marks]
    return marks