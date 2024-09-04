# Chatbot Model with GRU and LSTM

This repository contains code for training a chatbot model using a combination of GRU (Gated Recurrent Unit) and LSTM (Long Short-Term Memory) layers. The model is designed to classify user inputs into predefined intents and provide appropriate responses.You can also use Albert for sequence classification for that.

## Project Overview

The project involves:
- Loading and preparing training data from a JSON file.
- Tokenizing text data and using pre-trained GloVe word embeddings.
- Building and training a deep learning model using TensorFlow and Keras.
- Saving the trained model, tokenizer, and label encoder for future use.

## Files

- `intentfile.json`: Contains the training data with intents and responses.
- `glove.6B.100d.txt`: Pre-trained GloVe embeddings (must be placed in the same directory).
- `chatbot_model_advanced_GRU_LSTM.h5`: The trained model file.
- `tokenizer.json`: Tokenizer configuration.
- `label_encoder.json`: Label encoder configuration.

## Requirements

- TensorFlow
- NumPy
- Scikit-learn

You can install the required Python packages using:

```bash
pip install tensorflow numpy scikit-learn
