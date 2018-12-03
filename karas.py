from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.optimizers import Adam

# Using keras to load the dataset with the top_words
top_words = 10000
(X_train, y_train), (_,_) = imdb.load_data(num_words=top_words)

# Pad the sequence to the same length
max_review_length = 1600
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

# Using embedding from Keras
embedding_vecor_length = 300
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.6))
model.add(Dense(180,activation='sigmoid', kernel_regularizer=regularizers.l2(0.02)))
model.add(Dropout(0.6))
model.add(Dense(1,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=12, validation_split=0.2,callbacks=[tensorBoardCallback], batch_size=32)

# model.save("trained_demo.h5")
model.load_weights('trained_demo.h5')


from nltk import word_tokenize
import nltk
from keras.preprocessing import sequence
import numpy as np
word_index = imdb.get_word_index()

# words = 'i hate this movie'
# test = np.array([word_index[word] if word in word_index else 0 for word in words])

# test=sequence.pad_sequences([test],maxlen=max_review_length)
# print(model.predict(test))

words = ' They ensure it’s a solitary position to leave her on an island, set strict occupational rules for her to break, and very deliberately lead their antagonist through a series of meticulously designed pylons of human shaped food for entertainment. It’s factory-line horror as predictable as it is bland.'
test = np.array([word_index[word] if word in word_index else 0 for word in words])

test=sequence.pad_sequences([test],maxlen=max_review_length)
print(model.predict(test)[0])

# words = 'this movie is okay'
# test = np.array([word_index[word] if word in word_index else 0 for word in words])

# test=sequence.pad_sequences([test],maxlen=max_review_length)
# print(model.predict(test))