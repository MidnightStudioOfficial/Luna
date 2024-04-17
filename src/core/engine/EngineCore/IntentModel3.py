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

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


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

torch.manual_seed(42)
np.random.seed(42)

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



import torchtext.vocab as vocab
import torch
import torch.nn as nn
from transformers import BertModel

class NeuralNetwork2(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNetwork2, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc1(pooled_output)
        return logits


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, bert_finetune=False):
        super(NeuralNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        if not bert_finetune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # LSTM
        lstm_output, _ = self.lstm(pooled_output.unsqueeze(1))  # Add a dimension for the channel
        lstm_output = lstm_output.squeeze(1)  # Remove the added channel dimension
        lstm_output = self.dropout(lstm_output)

        # CNN
        cnn_input = pooled_output.unsqueeze(1)  # Add a dimension for the channel
        cnn_output = self.conv1d(cnn_input)
        cnn_output = self.maxpool(cnn_output)
        cnn_output = self.dropout(cnn_output.squeeze(-1))

        # Concatenate LSTM and CNN outputs
        combined_output = torch.cat((lstm_output, cnn_output), dim=1)

        logits = self.fc1(combined_output)
        return logits


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

        # Load pre-trained model and tokenizer
        print(os.path.realpath('Data/intent_model'))
        if os.path.exists('Data/intent_model') and os.path.exists('Data/label_encoder.pickle'):
            # Load pre-trained model and tokenizer
            self.model = BertForSequenceClassification.from_pretrained('Data/intent_model')
            self.tokenizer = BertTokenizer.from_pretrained('Data/intent_model')
            with open('Data/label_encoder.pickle', 'rb') as enc:
                self.label_encoder = pickle.load(enc)
        else:
            self.testing = ["what do you want to do?", "I am bored", "What is your favorite holiday?", "hi", "can you tell me a joke"]
            self.testing2 = ["what do you want to do?", "I am bored", "What is your favorite holiday?", "hi", "can you tell me a joke", "can you get the weather", "can you play me some music"]
            self.wordnet_lemmatizer = WordNetLemmatizer()
            self.stop_words_eng = set(stopwords.words('english'))
            self.stop_words_eng.remove("what")
            self.train2()

    def getIntent(self, utterance: str):
        """
        Predicts the intent of an utterance using the BERT-based model.

        Parameters:
            utterance (str): The utterance entered by the user.

        Returns:
            dict: A dictionary containing the following key-value pairs:
                'intent' (str): The predicted intent.
                'probability' (float): The probability score for the predicted intent.
        """
        # Tokenize the utterance
        inputs = self.tokenizer(utterance, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get predictions from the model
        result = self.model(input_ids, attention_mask)
        probabilities = torch.softmax(result.logits, dim=1)
        max_probability, predicted_label = torch.max(probabilities, dim=1)

        # Decode the predicted label
        predicted_label = predicted_label.item()
        predicted_intent = self.label_encoder.inverse_transform([predicted_label])[0]

        # Get additional information from the skill associated with the predicted intent
        skill = self.skills.skills[predicted_intent]
        entities = skill.parseEntities(self.nlp(utterance))

        # Generate response based on predicted intent and entities
        response = skill.actAndGetResponse(intentCheck=[predicted_intent], skills=self.skills.skills, **entities)

        return {'intent': predicted_intent, 'probability': max_probability.item(), 'response': response}

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
        print("Training neural network with BERT...")

        # Load BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(labels))

        # Tokenize training data
        sentences = [sample for skill in s.skills.values() for sample in skill.samples]
        labels = [skill_name for skill_name in s.skills.keys() for _ in s.skills[skill_name].samples]
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        #labels = torch.tensor([self.label_encoder.transform([label])[0] for label in labels])
        labels = torch.tensor(self.label_encoder.transform(labels), dtype=torch.long)

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)

        # Define optimizer and loss function
        optimizer = AdamW(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        model.train()
        for epoch in range(400):  # Adjust the number of epochs as needed 5 10000
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{5}, Average Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}")

        print("Model trained.")

        # Save the trained model
        model.save_pretrained("Data/intent_model")
        tokenizer.save_pretrained("Data/intent_model")
        with open('Data/label_encoder.pickle', 'wb') as ecn_file:
            pickle.dump(self.label_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
        self.model = model
        self.tokenizer = tokenizer
        #self.label_encoder = label_encoder

    def preprocess_sentence2(self, sentence):
        sentence = sentence.lower()
        punctuations = "?:!.,;'`´"
        sentence_words = word_tokenize(sentence)
        stemmed_sentence = [self.porter_stemmer.stem(word) for word in sentence_words if word not in self.stop_words_eng and word not in punctuations]
        return " ".join(stemmed_sentence)

  