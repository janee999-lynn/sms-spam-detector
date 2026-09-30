"""
Retrain script - run this once with: python retrain.py
Regenerates model.h5, tokenizer.pkl, and config.pkl using YOUR installed
TensorFlow version, with the final architecture (Early Stopping, fixed seed).
"""

import pandas as pd
import numpy as np
import pickle
import random
import os

os.environ['PYTHONHASHSEED'] = '7'
import tensorflow as tf
tf.random.set_seed(7)
np.random.seed(7)
random.seed(7)

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

print("Loading dataset...")
df = pd.read_csv("dataset/A2_sms_spam_detection.csv")

X = np.array(df['text'].astype(str).tolist(), dtype=object)
y = np.array(df['label'].map({'ham': 0, 'spam': 1}).tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vocab_size = 1000
max_len = 20
embedding_dim = 64
rnn_units = 64

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

print("Building model...")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    SimpleRNN(rnn_units),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training (with Early Stopping)...")
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)
print(f"\nTraining stopped after {len(history.history['loss'])} epochs.")

loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test accuracy: {acc:.2%}")

print("Saving model, tokenizer, and config...")
model.save("model/model.h5")
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("model/config.pkl", "wb") as f:
    pickle.dump({"max_len": max_len}, f)

print("\nDone! model.h5, tokenizer.pkl, and config.pkl have been refreshed.")
print("Now try running: streamlit run app.py")
