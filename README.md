# Emoji Predictor using LSTM and GloVe Embeddings

This repository contains code for an Emoji Predictor implemented using LSTM (Long Short-Term Memory) neural networks and GloVe (Global Vectors for Word Representation) word embeddings.

## Dataset
- The dataset comprises two CSV files: `train_emoji.csv` for training and `test_emoji.csv` for testing.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `emoji`, `tensorflow`

## Usage
1. Download the GloVe embeddings from [this link](https://www.kaggle.com/datasets/watts2/glove6b50dtxt) and place the `glove.6B.50d.txt` file in the repository.
2. Run the code in a Python environment with the necessary dependencies installed.
3. The code reads the training and testing data from the provided CSV files and implements an LSTM-based model for emoji prediction.
4. The model is trained and evaluated using the provided dataset, and predictions are made for the test set.

## Code Explanation
- The code first loads the dataset and GloVe word embeddings.
- It preprocesses the text data and creates input-output pairs for the LSTM model.
- The LSTM model architecture consists of two LSTM layers followed by dropout layers and a softmax output layer.
- The model is compiled with the Adam optimizer and categorical cross-entropy loss.
- Training is performed for 40 epochs with a batch size of 32.
- Evaluation metrics such as accuracy are computed on the test dataset.
- Predictions are made on the test set, displaying the actual emoji and the predicted emoji for sample test cases.

## Additional Notes
- The provided code demonstrates a basic implementation of an LSTM-based Emoji Predictor using word embeddings.
- Adjustments to hyperparameters, model architecture, or dataset preprocessing can be made for further improvements.

