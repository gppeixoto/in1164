import numpy as np
import pandas as pd
import spacy
from keras.layers import Dense, Input, Conv1D, Bidirectional, LSTM, Embedding, Permute, Dropout, \
    concatenate, multiply, subtract, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import train_test_split

############################################################
#################### Data Loading ##########################
############################################################

df = pd.read_csv("./quora_duplicate_questions.tsv", sep='\t', encoding='utf-8')
df = df[~df.question1.isnull() & ~df.question2.isnull() & ~df.is_duplicate.isnull()]
nlp = spacy.load("en")

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

data1 = np.load("data1-50.npy")
data2 = np.load("data2-50.npy")
assert data1.shape == data2.shape

############################################################
###################### Model ###############################
############################################################

input1 = Input(shape=(50,), dtype='int32', name='one_input')
input2 = Input(shape=(50,), dtype='int32', name='two_input')

embs = get_embeddings(nlp.vocab)

embedding_layer = Embedding(
    input_dim=embs.shape[0], 
    output_dim=300, 
    input_length=50,
    trainable=False, 
    weights=[embs], 
    name='embedding1'
)

x = embedding_layer(input1)
shared_lstm = Bidirectional(LSTM(150), name='shared_bilstm')
x = shared_lstm(x)

y = Embedding(
    input_dim=embs.shape[0], 
    output_dim=300, 
    input_length=50,
    trainable=False, 
    weights=[embs], 
    name='embedding2'
)(input2)

y = shared_lstm(y)

concat = concatenate([x, y])
mul = multiply([x, y])
sub = subtract([x, y])
concat = concatenate([concat, mul, sub])

concat = Dense(200, activation='relu')(concat)
concat = Dense(200, activation='relu')(concat)
concat = Dense(200, activation='relu')(concat)
concat = Dropout(.5)(concat)
output = Dense(2, activation='softmax', name='dense')(concat)
model = Model([input1, input2], output, name='deep_lstm')
model.compile("adam", "binary_crossentropy", metrics=['accuracy'])
model.summary()

from keras.utils import plot_model
# plot_model(model, to_file='attn-cnn-lstm.png', show_layer_names=True, show_shapes=True)

############################################################
################ Training and Eval #########################
############################################################

labels = np.load("labels.npy")
x1_train, x1_test, x2_train, x2_test, labels_train, labels_test = train_test_split(
    data1, data2, labels,
    test_size=0.15
)

N_EPOCHS = 10
history = model.fit(
    [x1_train, x2_train],
    labels_train,
    batch_size=128,
    validation_split=.15,
    epochs=N_EPOCHS,
    callbacks=[EarlyStopping(patience=2)]
)

preds = model.predict([x1_test, x2_test]).argmax(axis=-1)
y_true = labels_test.argmax(axis=-1)

print(prfs(y_true, preds))
print(accuracy_score(y_true, preds))
