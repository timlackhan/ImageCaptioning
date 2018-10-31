from keras.layers import Multiply,Permute,add, Embedding, TimeDistributed, RepeatVector, LSTM, GRU, concatenate , Input, Reshape, Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.models import load_model 
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from tqdm import tqdm
import numpy as np
import h5py as h5py
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def my_loadData(text_test_dir, image_test_dir):
    text = []
    images = []
    for i in range(150):
        readPath = image_test_dir + str(i+1) + ".png"
        img = np.array(Image.open(readPath))
        imgInfo = img.shape
        dstHeight = int(imgInfo[0]*0.25)
        dstWidth = int(imgInfo[1]*0.25)
        dst = cv2.resize(img,(dstWidth,dstHeight))/255.0
        images.append(dst)
    images = np.array(images, dtype=float)
    text_train = open(text_test_dir,"r")
    for i in range(150):
        line = text_train.readline().strip("\n")
        text.append(line)
    text_train.close()
    return images, text
	
image_test_dir = "/rap_blues/test_imgs/"
text_test_dir = "/rap_blues/test_labels2/test_labels2.txt"
train_features, texts = my_loadData(text_test_dir, image_test_dir)
tokenizer = Tokenizer(filters='', split=" ", lower=False)
tokenizer.fit_on_texts([load_doc('/rap_blues/my_bootstrap.vocab')])
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(texts)
max_sequence = max(len(s) for s in train_sequences)
max_length = 48

def preprocess_data(sequences, features):
    X, y, image_data = list(), list(), list()
    for img_no, seq in enumerate(sequences):
        for i in range(1, len(seq)):
            # Add the sentence until the current count(i) and add the current count to the output
            in_seq, out_seq = seq[:i], seq[i]
            # Pad all the input token sentences to max_sequence
            in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
            # Turn the output into one-hot encoding
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # Add the corresponding image to the boostrap token file
            image_data.append(features[img_no])
            # Cap the input sentence to 48 tokens and add it
            X.append(in_seq[-48:])
            y.append(out_seq)
    return np.array(X), np.array(y), np.array(image_data)

  
X, y, image_data = preprocess_data(train_sequences, train_features)

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(my_model, tokenizer, photo, max_length):
    photo = np.array([photo])
    # seed the generation process
    in_text = '<START> '
    # iterate over the whole length of the sequence
    print('\nPrediction---->\n\n<START> ', end='')
    for i in range(150):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = my_model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += word + ' '
        # stop if we predict the end of the sequence
        print(word + ' ', end='')
        if word == '<END>':
            break
    return in_text
	
###  baseline  ###
def attention_3d_block(inputs): 
        input_dim = int(inputs.shape[2]) 
        a = Permute((2, 1))(inputs) 
        a = Dense(48, activation='softmax')(a) 
        a_probs = Permute((2, 1), name='attention_vec')(a) 
        output_attention_mul = Multiply()([inputs, a_probs]) 
        return output_attention_mul 
    

image_model = Sequential()
image_model.add(Conv2D(16, (3,3), padding='valid', activation='relu', input_shape=(256, 256, 3,)))
image_model.add(Conv2D(16, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
image_model.add(Conv2D(32, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
image_model.add(Conv2D(64, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
image_model.add(Flatten())
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))
image_model.add(RepeatVector(max_length))
visual_input = Input(shape=(256, 256, 3,))
encoded_image = image_model(visual_input)

language_input = Input(shape=(max_length,))
language_model = Embedding(vocab_size, 50, input_length=max_length, mask_zero=False)(language_input)
language_model = GRU(128, return_sequences=True)(language_model)
language_model = GRU(128, return_sequences=True)(language_model)
#Create the decoder
decoder = concatenate([encoded_image, language_model])
decoder = attention_3d_block(decoder) 
decoder = GRU(512, return_sequences=True)(decoder)
decoder = GRU(512, return_sequences=False)(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder) 
# Compile the model
my_model = Model(inputs=[visual_input , language_input], outputs=decoder)
my_model.load_weights("/rap_blues/lunwen/baseline_attention/220/org-weights-epoch-0080--val_loss-0.6853--loss-0.0024.hdf5")

actual, predicted = list(), list()
for i in range(len(texts)):
    print(i)
    yhat = generate_desc(my_model, tokenizer, train_features[i], max_length)
    print('\n\nReal---->\n\n' + texts[i])
    actual.append([texts[i].split()])
    predicted.append(yhat.split())
	
def save_to_file(path, file):
    fh = open(path, 'a')
    fh.write(file)
    fh.close()

for i in range(len(predicted)):
    str=""
    for j in range(len(predicted[i])):
        str=str+predicted[i][j]+" "
    str=str[:-1]
    str=str+"\n"
    save_to_file("/rap_blues/lunwen/baseline_attention/predicted.txt", str)
	
bleu = corpus_bleu(actual, predicted)
bleu