# Simple chatbot for day-to-day dialogs

Chatbot was written using full transformer model with 6 encoders and 6 decoders. Dataset was created using simple_dialogs dataset. Decoding method is nucleus sampling.


## Demo

<img src="gifs/chatting.gif" width="768px" />

## TODO
Add memory to model.
## Usage
- for training model use `train.py`
- for talking with chatbot use `test.py`
- trained model is in `saves/` directory
