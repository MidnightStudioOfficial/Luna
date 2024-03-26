"""
Conversational Engine using a Neural Network

This script defines a conversational engine that can predict the intent of an utterance using a neural network.
The engine uses deep learning techniques to classify user input into predefined intents, enabling it to respond
with appropriate actions or responses. It leverages libraries such as TensorFlow, Keras, NLTK, and spaCy for NLP
functionalities and model training.

The Engine2 class includes methods for intent prediction, data preprocessing, model training, and skill management.
It utilizes labeled training data with corresponding intents to train a neural network and uses a Tokenizer for text
preprocessing. The engine also incorporates built-in skills for handling various user requests.

Note: This implementation is part of a larger project and may have specific dependencies on the rest of the codebase.
"""
# Standard libraries
import os  # For interacting with the operating system (e.g., checking file existence)
import pickle  # For pickling (serializing) Python objects

# Third-party libraries
from numpy import max, argmax, array as np_max, argmax, array  # NumPy for numerical operations and array handling
from nltk import word_tokenize  # Natural Language Toolkit for NLP functionalities
from nltk.corpus import stopwords  # NLTK's stopwords for filtering common words
from nltk.stem import WordNetLemmatizer  # WordNetLemmatizer for word lemmatization
from spacy import load as spacy_load  # spaCy for advanced natural language processing

# TensorFlow and Keras libraries
from tensorflow import keras  # TensorFlow library for deep learning
from keras.preprocessing.text import Tokenizer  # Tokenizer for text preprocessing
from keras.preprocessing.sequence import pad_sequences  # Padding sequences for input data


# PyTorch libraries
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import numpy as np


# Third-party library
from sklearn.preprocessing import LabelEncoder  # LabelEncoder for encoding target labels
from sklearn.model_selection import train_test_split  # train_test_split for splitting data into training and testing sets
from nltk.stem import PorterStemmer  # PorterStemmer for word stemming (reducing words to their base or root form)

# Local import
from core.skill.bulitin_skills import BuiltinSkills  # Custom built-in skills for the conversational engine

class IntentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to torch.long
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), self.labels[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = self.pooling(embedded.permute(0, 2, 1)).squeeze(2)
        x = torch.relu(self.fc1(pooled))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# This class defines a conversational engine that can predict the intent of an utterance using a neural network.
class IntentModel():
    # The constructor takes in several optional arguments to customize the behavior of the engine.
    def __init__(self, lemmatize_data=True, filepath=None, modelpath=None):
        """
        app: any object | usually the chatbot object
        lemmatize_data: bool | True includes lemmatization, False excludes it
        filepath: str | the path to the .csv file containing the training data
        modelpath str, optional | the path to the .p file containing a pickled model you wish to use. If passed, will use that model instead of retraining from the training data. This leads to faster instantiation.
        """
        # Define parameters
        self.max_len = 25

        # Load built-in skills for the engine
        self.skills = BuiltinSkills()

        # Load the spaCy language model for natural language processing
        self.nlp = spacy_load("en_core_web_sm")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # WordNetLemmatizer is used to reduce words to their base or root form (lemmas).
        # It helps to normalize different forms of a word, such as plurals or verb conjugations, to a common base form.
        self.wordnet_lemmatizer = WordNetLemmatizer()

        # Create an instance of the PorterStemmer class for stemming words.
        # The PorterStemmer is used to reduce words to their base or root form, which can help in information retrieval or text analysis tasks.
        self.porter_stemmer = PorterStemmer()

        # Initialize the set of English stopwords, which are common words that are usually removed from text during preprocessing.
        # Stopwords are words like "the", "and", "is", "are", etc., that do not contribute much to the meaning of the text.
        self.stop_words_eng = set(stopwords.words('english'))
        self.stop_words_eng.remove("what")

        if os.path.exists('Data/sir-bot-a-lot.brain') and os.path.exists('Data/tokenizer.pickle') and os.path.exists('Data/label_encoder.pickle'):
            self.model = torch.load('Data/sir-bot-a-lot.brain')
            with open('Data/tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            with open('Data/label_encoder.pickle', 'rb') as enc:
                self.label_encoder = pickle.load(enc)
        else:
            self.testing = ["what do you want to do?", "I am bored", "What is your favorite holiday?", "hi", "can you tell me a joke"]
            self.testing2 = ["what do you want to do?", "I am bored", "What is your favorite holiday?", "hi", "can you tell me a joke", "can you get the weather", "can you play me some music"]
            self.wordnet_lemmatizer = WordNetLemmatizer()
            self.stop_words_eng = set(stopwords.words('english'))
            self.stop_words_eng.remove("what")
            self.train2()

    def getIntent(self, utterance):
        """
        Predicts the intent of an utterance using the neural network.

        Parameters:
            utterance (str): The utterance entered by the user.

        Returns:
            dict: A dictionary containing the following key-value pairs:
                  'intent' (str): The predicted intent.
                  'probability' (float): The probability score for the predicted intent.
        """
        result = self.model(torch.tensor(self.tokenizer.texts_to_sequences([utterance])))
        tag = self.label_encoder.inverse_transform([torch.argmax(result).item()])
        params = {'intentCheck': tag, 'skills': self.skills.skills}
        skill = self.skills.skills[tag[0]]
        params |= skill.parseEntities(self.nlp(utterance))
        response = skill.actAndGetResponse(**params)
        # For PyTorch tensors:
        max_probability, _ = torch.max(result, dim=1)
        print(response + " " + str(max_probability.item()))
        return {'intent': response, 'probability': max_probability}
    def preprocess_sentence(self, sentence):
        sentence = sentence.lower()
        punctuations = "?:!.,;'`´"
        sentence_words = word_tokenize(sentence)
        lemmatized_sentence = []

        for word in sentence_words:
            #if word in stop_words_eng:
            #     continue
            if word in punctuations:
                continue
            lemmatized_word = self.wordnet_lemmatizer.lemmatize(word, pos="v")
            lemmatized_sentence.append(lemmatized_word)

        return " ".join(lemmatized_sentence)

    def preprocess_sentence2(self, sentence):
        """
        Preprocesses a sentence using stemming and removes stopwords and punctuations.

        Parameters:
            sentence (str): The input sentence to preprocess.

        Returns:
            str: The preprocessed sentence after stemming and removing stopwords and punctuations.
        """
        # Convert the sentence to lowercase for consistency
        sentence = sentence.lower()

        # Define a string of punctuations to ignore
        punctuations = "?:!.,;'`´"

        # Tokenize the sentence into individual words
        sentence_words = word_tokenize(sentence)

        # Initialize a list to store stemmed words
        stemmed_sentence = []

        # Loop through each word in the tokenized sentence
        for word in sentence_words:
            # Skip the word if it is a common English stopword
            if word in self.stop_words_eng:
                continue

            # Skip the word if it is a punctuation
            if word in punctuations:
                continue

            # Apply stemming to the word to get the root form
            stemmed_word = self.porter_stemmer.stem(word)
            stemmed_sentence.append(stemmed_word)

        # Return the preprocessed sentence as a string
        return " ".join(stemmed_sentence)

    def train2(self):
        print("Loading all skills...")
        s = self.skills
        training_sentences = []
        training_labels = []
        labels = []
        for intent, skill in s.skills.items():
            for sample in skill.samples:
                training_sentences.append(sample)
                training_labels.append(intent)
            if intent not in labels:
                labels.append(intent)
        num_classes = len(labels)
        lemmatized_training_sentences = [self.preprocess_sentence(sentence) for sentence in training_sentences]
        lbl_encoder = LabelEncoder()
        lbl_encoder.fit(training_labels)
        training_labels = lbl_encoder.transform(training_labels)
        vocab_size = 2000
        embedding_dim = 62
        max_len = 25
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(lemmatized_training_sentences)
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(lemmatized_training_sentences)
        padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
        X_train, X_temp, y_train, y_temp = train_test_split(padded_sequences, np.array(training_labels), test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        dataset_train = IntentDataset(X_train, y_train)
        dataset_val = IntentDataset(X_val, y_val)
        dataset_test = IntentDataset(X_test, y_test)
        dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=32)
        dataloader_test = DataLoader(dataset_test, batch_size=32)
        model = NeuralNetwork(vocab_size, embedding_dim, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        epochs = 10000 #1000
        for epoch in range(epochs):
            model.train()
            for inputs, labels in dataloader_train:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for inputs, labels in dataloader_val:
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    val_accuracy += (torch.argmax(outputs, dim=1) == labels).float().mean().item()
                val_loss /= len(dataloader_val)
                val_accuracy /= len(dataloader_val)
                print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f} Loss: {loss:.4f}")
        test_loss = 0.0
        test_accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader_test:
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                test_accuracy += (torch.argmax(outputs, dim=1) == labels).float().mean().item()
                test_loss /= len(dataloader_test)
            test_accuracy /= len(dataloader_test)
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        torch.save(model, "Data/sir-bot-a-lot.brain")
        with open('Data/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('Data/label_encoder.pickle', 'wb') as ecn_file:
            pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model trained and saved.")
        self.model = model
        self.tokenizer = tokenizer

    def preprocess_sentence2(self, sentence):
        sentence = sentence.lower()
        punctuations = "?:!.,;'`´"
        sentence_words = word_tokenize(sentence)
        stemmed_sentence = [self.porter_stemmer.stem(word) for word in sentence_words if word not in self.stop_words_eng and word not in punctuations]
        return " ".join(stemmed_sentence)

  