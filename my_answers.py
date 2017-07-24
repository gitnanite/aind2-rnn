import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import keras


# Fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    nbpairs = len(series) - window_size
		
    # containers for input/output pairs
    X = []
    y = []
	
    for xpair in range(nbpairs):
        X.append(series[xpair:xpair+window_size])
		
    y = series[window_size:]
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# Build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1))
	
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    text = text.replace('"',' ')
    text = text.replace('$',' ')
    text = text.replace('%',' ')
    text = text.replace('&',' ')
    text = text.replace("'",' ')
    text = text.replace('(',' ')
    text = text.replace(')',' ')
    text = text.replace('*',' ')
    text = text.replace('-',' ')
    text = text.replace('/',' ')
    text = text.replace('0',' ')
    text = text.replace('1',' ')
    text = text.replace('2',' ')
    text = text.replace('3',' ')
    text = text.replace('4',' ')
    text = text.replace('5',' ')
    text = text.replace('6',' ')
    text = text.replace('7',' ')
    text = text.replace('8',' ')
    text = text.replace('9',' ')
    text = text.replace('@',' ')
    text = text.replace('\xa0',' ')
    text = text.replace('ã¢','a')
    text = text.replace('ã¨','e')
    text = text.replace('ã©','e')
    text = text.replace('ã','a')
    text = text.replace('à','a')
    text = text.replace('é','a')
    text = text.replace('è','e')
    text = text.replace('â','a')
	
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']
    start_lcase_hex = 0x61
    end_lcase_hex = 0x7B
    lower_case_hex = [x for x in range(start_lcase_hex,end_lcase_hex)]
    lower_case_chr = [chr(l) for l in lower_case_hex ]
    desired_characters = punctuation + lower_case_chr 
    #in case I missed something, I just erase anything unwanted
    ''.join([i for i in text if (i in desired_characters)])
    #punctuation = ['!', ',', '.', ':', ';', '?']
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    nbpairs = len(text) - window_size
	
    # containers for input/output pairs
    inputs = []
    outputs = []
		
    for xpair in range(0,nbpairs,step_size):
        inputs.append(text[xpair:xpair+window_size])
        outputs.append(text[xpair+ window_size])
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
	
    return model
