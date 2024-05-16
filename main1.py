import numpy as np
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Dropout, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('BBC News Text.csv')

# Text preprocessing
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(preprocess_text)

# Tokenization and padding
MAX_SENTENCE_LENGTH = 50
MAX_SENTENCES = 15
MAX_WORDS = 10000

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['clean_text'])

X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, padding='post', truncating='post')

# Splitting into train and test sets
y = pd.get_dummies(df['category']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hierarchical Attention Network architecture
class HierarchicalAttention(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.attention_dim), initializer='normal')
        self.b = self.add_weight(name='b', shape=(self.attention_dim,), initializer='normal')
        self.u = self.add_weight(name='u', shape=(self.attention_dim, 1), initializer='normal')
        super(HierarchicalAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.bias_add(K.dot(x, self.W), self.b)
        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Model architecture
sentence_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32')
embedded_sequences = Embedding(MAX_WORDS, 100, input_length=MAX_SENTENCE_LENGTH)(sentence_input)
lstm_word = Bidirectional(LSTM(50, return_sequences=True))(embedded_sequences)
word_attention = HierarchicalAttention(100)(lstm_word)

sentence_encoder = Model(sentence_input, word_attention)

review_input = Input(shape=(MAX_SENTENCES, MAX_SENTENCE_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentence_encoder)(review_input)
lstm_sentence = Bidirectional(LSTM(50, return_sequences=True))(review_encoder)
sentence_attention = HierarchicalAttention(100)(lstm_sentence)

output = Dense(5, activation='softmax')(sentence_attention)
model = Model(review_input, output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=64, callbacks=[early_stopping])

print(classification_report(y_test, y_pred))

