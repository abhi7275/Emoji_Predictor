import numpy as np
import pandas as pd
import emoji as emoji
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential
emoji_dictionary = {"0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                    "5": ":hundred_points:",
                    "6": ":fire:",
                    "7": ":face_blowing_a_kiss:",
                    "8": ":chestnut:",
                    "9": ":flexed_biceps:",
                    "10": ":grinning_face:",
                   }


train = pd.read_csv('train_emoji.csv',header=None)
test = pd.read_csv('test_emoji.csv',header=None)
#
# data = train.values
# for i in range(10):
#   print(data[i][0],emoji.emojize(emoji_dictionary[str(data[i][1])]))
#Glove is large thats why doesn't uploaded in this file \
#we can download from this link - https://www.kaggle.com/datasets/watts2/glove6b50dtxt
embeddings = {}
with open('glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float')
        embeddings[word] = coeffs
# print(embeddings['eat'])
# print(embeddings['play'])

X_train = train[0]
X_test = test[0]

Y_train = to_categorical(train[1])
Y_test = to_categorical(test[1])

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test)


def getOutputEmbeddings(X):
    embedding_matrix_output = np.zeros((X.shape[0], 10, 50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]

    return embedding_matrix_output


emb_X_train = getOutputEmbeddings(X_train)
emb_X_test = getOutputEmbeddings(X_test)
# print(emb_X_train.shape)
# print(emb_X_test.shape)

model = Sequential()
model.add(LSTM(64,input_shape=(10,50),return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(64,input_shape=(10,50)))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(emb_X_train,Y_train,batch_size=32,epochs=40,shuffle=True,validation_split=0.1)

model.evaluate(emb_X_test,Y_test)

pred = model.predict_classes(emb_X_test)
for i in range(10):
    print(''.join(X_test[i]))
    print(emoji.emojize(emoji_dictionary[str(np.argmax(Y_test[i]))]))
    print(emoji.emojize(emoji_dictionary[str(pred[i])]))