import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy;
nlp=spacy.load('en')
# from gensim.models import Word2Vec
# from utils import *

# w2vModel = Word2Vec.load('Essayw2v.model')
# vocab = set(w2vModel.wv.vocab)
# dic = w2vModel.wv
# w2v = dict(zip(w2vModel.wv.index2word, w2vModel.wv.syn0))
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
model = load_model('EssayModelRNN.h5')

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
    marks = model.predict(essays)
    # print(marks)
    marks = (marks-5)*1.4+5
    marks = [x[0] for x in marks]
    # print(marks)
    marks = [min(x,10) for x in marks]
    marks = [max(x,0) for x in marks]
    # marks = [int(round(x)) for x in marks]
    print(marks)
    return marks

essay='European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices. by 2020, my name is John. I used to live in Turkey in Indonesia before January'
essay2="Last month a grand exhibition was held in our city. My friends and I went to see it in evening. Our first impression on entering the grounds was that whole thing looked like a fairyland. The vast space was decorated in magnificent, bright and purple colour and lit up with countless lights. Men, women and children were moving from corner to corner, admiring the beauty of all kinds of stalls set up. These stalls were like small shops. While the stalls made a very interesting sight, what attracted us most was the Children's Corner in the exhibition. The Children's Corner was crowded with boys and girls. All types of amusements could be seen here. Children and some grown-ups were enjoying the giant wheel, wooden hoses, dodge-cars, railway train and other things. I too had my share of fun with my friends and returned home after enjoying a most delightful evening. "
essay3='by 2020, my name is John. I used to live in Turkey in Indonesia before January'

getmarks([essay,essay2,essay3])