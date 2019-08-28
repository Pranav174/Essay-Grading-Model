import pandas as pd
import numpy as np
import re
def getData():
    data = pd.read_csv('./asap-aes/training_set_rel3.tsv', sep='\t', encoding = "latin")
    data['illegible']=data.essay.str.contains('(\?\?\?|illegible|not legible)')
    data = data[data.illegible==False]
    data = data.drop('illegible', 1)
    data['essay'] = data['essay'].apply(lambda x: re.sub(' +', ' ', x.strip()))
    return data

def replaceNER(essay):
    essay = re.sub("@person[^ ]*","@person",essay) 
    essay = re.sub("@month[^ ]*","@date",essay) 
    essay = re.sub("@location[^ ]*","@location",essay) 
    essay = re.sub("@num[^ ]*","@num",essay) 
    essay = re.sub("@organization[^ ]*","@organization",essay) 
    essay = re.sub("@date[^ ]*","@date",essay) 
    essay = re.sub("@percent[^ ]*","@percent",essay) 
    essay = re.sub("@money[^ ]*","@money",essay) 
    essay = re.sub("@time[^ ]*","@time",essay) 
    essay = re.sub("@dr[^ ]*","@person",essay) 
    essay = re.sub("@caps[^ ]*","@caps",essay)
    return essay

def basicClean(essay):
    essay = essay.lower()
    essay = re.sub("n't"," not",essay)
    essay = re.sub("'re"," are",essay)
    essay = re.sub("it's","it is",essay)
    essay = re.sub("'ve'"," have",essay)
    return essay