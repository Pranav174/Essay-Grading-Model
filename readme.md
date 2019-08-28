# Highesh KAPPA Score:
 0.82 using 1-dimentional convolution over embeddings

# Notebook's Purpose:
* FullScores.ipynb - gather different domain scores of different essay sets
* FullFeatures.ipynb - Manual feature extractions like word count, depth of sentences etc.
* Glove.ipynb - preprocessing and implementation with glove embedding
* word2vecEmbedding.ipynb - Generating word2vec model for word embeddings
* Regression.ipynb - Simple regression Model
* Convolutional.ipynb - Use of 1-dimentional convolution in final model (this gave the best score)
* Full_scores_features.ipynb - Model with multiple inputs (extracted features) and multiple outputs (domain scores) to train on .
# Top Model's Summary (without feature extraction & multi-grading):

```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 1200, 300)         12733200  
_________________________________________________________________
conv1d_7 (Conv1D)            (None, 1196, 64)          96064     
_________________________________________________________________
max_pooling1d_7 (MaxPooling1 (None, 34, 64)            0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 2176)              0         
_________________________________________________________________
dropout (Dropout)            (None, 2176)              0         
_________________________________________________________________
dense_14 (Dense)             (None, 64)                139328    
_________________________________________________________________
dense_15 (Dense)             (None, 1)                 65        
=================================================================
Total params: 12,968,657
Trainable params: 235,457
Non-trainable params: 12,733,200
_________________________________________________________________
```

# How to use:

I have also implemented files to use the model as an api to get essay's score

Load the essay checker python file
```sh
from essayChecker import *
```

Then use the getmarks() method
```sh
Predicted_marks = getmarks([list of esssays to be checked])
```