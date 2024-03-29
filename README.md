# Simple chatbot for day-to-day dialogs

Chatbot was written using full transformer model with 6 encoders and 6 decoders. Dataset was created using `simple_dialogs` dataset. Decoding method is nucleus sampling.


## Demo

<img src="gifs/chatting.gif" width="768px" />


## Usage
- for training model use `train.py`
- for talking with chatbot use `test.py`
- trained model is in `saves/` directory

## Version change
- V1 - Initial chatbot using vanilla transformer without memory
- V2 - Chatbot with memory using Transformer-XL
- V3 - Chatbot with longer memory using Compressive Transformer

## Note
Transformer with memory requires related dialogs to be in sequence. Due to the dataset used, sequences does not have one length, forcing to create empty dialogs which hurts the model performance. Also, this dataset when used as sequences has low amount of samples. For best experience use version 1. 
