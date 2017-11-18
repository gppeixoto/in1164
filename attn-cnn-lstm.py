import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import accuracy_score as acc
import spacy
from keras.layers import Dense, Input, Conv1D, Bidirectional, LSTM, Embedding, Permute, Dropout, \
    concatenate, merge
from keras.models import Model
from keras.optimizers import Adam
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

############################################################
#################### Data Loading ##########################
############################################################

df = pd.read_csv("./quora_duplicate_questions.tsv", sep='\t', encoding='utf-8')
df = df[~df.question1.isnull() & ~df.question2.isnull() & ~df.is_duplicate.isnull()]

nlp = spacy.load("en")
ones = df.question1.apply(nlp)
twos = df.question2.apply(nlp)

def get_features(docs, max_length):
    Xs = np.zeros((len(list(docs)), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc[:max_length]):
            Xs[i, j] = token.rank if token.has_vector else 0
    return Xs

def get_embeddings(vocab):
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank+1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors

data1 = get_features(ones, max_length=30)
data2 = get_features(twos, max_length=30)
assert data1.shape == data2.shape

############################################################
###################### Model ###############################
############################################################

input1 = Input(shape=(30,), dtype='int32', name='one_input')
input2 = Input(shape=(30,), dtype='int32', name='two_input')

# shared embedding layer
embs = get_embeddings(nlp.vocab)
embedding_layer = Embedding(
    input_dim=embs.shape[0], output_dim=300, input_length=30, 
    trainable=False, weights=[embs], name='embedding1')

x = embedding_layer(input1)

def per_word_attn_vector(inputs, suffix="1"):
    a = Permute((2, 1), name='permute_{}'.format(suffix))(inputs)
    attention_probs = Dense(30, activation='softmax', name='attnprobs_{}'.format(suffix))(a)
    attention_probs = Permute((2, 1), name='permuteback_{}'.format(suffix))(attention_probs)
    attention_vec = merge([inputs, attention_probs], mode='mul', name='attnlayer_{}'.format(suffix))
    return attention_vec

x = Conv1D(50, 3, name='conv1')(per_word_attn_vector(x))
x = Bidirectional(LSTM(50, name='lstm1'), name='bidirectional_lstm1')(x)
x = Dropout(.5, name='dropout1')(x)

y = embedding_layer = Embedding(
    input_dim=embs.shape[0], output_dim=300, input_length=30, 
    trainable=False, weights=[embs], name='embedding2')(input2)
y = Conv1D(50, 3, name='conv2')(per_word_attn_vector(y, suffix="2"))
y = Bidirectional(LSTM(50, name='lstm'), name='bidirectional_lstm2')(y)
y = Dropout(.5, name='dropout2')(y)

concat = concatenate([x, y])

output = Dense(2, activation='softmax', name='dense')(concat)
model = Model([input1, input2], output, name='deep_cnn_lstm')
model.compile(Adam(lr=0.01), 'categorical_crossentropy', metrics=['accuracy'])

from keras.utils import plot_model
plot_model(model, to_file='attn-cnn-lstm.png', show_layer_names=True, show_shapes=True)

############################################################
################ Training and Eval #########################
############################################################

N_EPOCHS = 10
print "word-based attention mechanism bef. convolution"
labels = to_categorical(df.is_duplicate)
history = model.fit(
    [data1, data2],
    labels,
    validation_split=.1,
    epochs=N_EPOCHS,
    callbacks=[EarlyStopping(patience=2)]
)
