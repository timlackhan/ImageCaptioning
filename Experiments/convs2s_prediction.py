from keras.layers import Dot, Multiply, add, Add, Embedding, TimeDistributed, RepeatVector, LSTM, GRU, concatenate , Input, Reshape, Dense,Activation,Dropout,Flatten,Conv1D, Conv2D,MaxPooling2D
from keras.layers.core import Activation,Lambda
from keras.backend import transpose, zeros, zeros_like
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
from keras.layers import Permute,merge


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
	
def save_to_file(path, file):
    fh = open(path, 'a')
    fh.write(file)
    fh.close()

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
        yhat = my_model.predict([photo, sequence, np.array(np.zeros((1, 2, 21)))], verbose=0)
        # convert probability to integer
        yhat = yhat[0][-1][:]
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
  
  
   
#text preprocessing   
image_test_dir = "/rap_blues/test_imgs/"
text_test_dir = "/rap_blues/test_labels2/test_labels2.txt"
train_features, texts = my_loadData(text_test_dir, image_test_dir)
tokenizer = Tokenizer(filters='', split=" ", lower=False)
tokenizer.fit_on_texts([load_doc('/rap_blues/my_bootstrap.vocab')])
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(texts)
max_sequence = max(len(s) for s in train_sequences)
max_length = 48


#encoder kernel=3
image_input = Input(shape=(256, 256, 3,))
image_conv1 = Conv2D(16, (3,3), padding='valid', activation='relu')(image_input)
image_conv2 = Conv2D(16, (3,3), activation='relu', padding='same', strides=2)(image_conv1)
image_conv3 = Conv2D(32, (3,3), activation='relu', padding='same', strides=2)(image_conv2)
image_conv4 = Conv2D(32, (3,3), activation='relu', padding='same', strides=2)(image_conv3)
image_conv5 = Conv2D(21, (3,3), activation='relu', padding='same', strides=2)(image_conv4)
image_encoder = Reshape((256,21))(image_conv5)

#decoder 
language_input = Input(shape=(max_length,))
padding_input = Input(shape=(2, 21, ))
language_model = Embedding(vocab_size, 21, input_length=max_length, mask_zero=False)(language_input)
padding_language = concatenate([padding_input, language_model], axis=1)
#1st block
decoder_conv = Conv1D(21, 3, padding='valid')(padding_language)
decoder_gate = Conv1D(21, 3, padding='valid', activation='sigmoid')(padding_language)
decoder_glu =  Multiply()([decoder_conv, decoder_gate]) 
decoder_1 = Add()([language_model, decoder_glu])
#attention
attention_matrix = Dot(axes=2)([decoder_1, image_encoder])
attention_softmax = Activation('softmax')(attention_matrix)
image_encoderTrans = Reshape((21,256))(image_encoder)
decoder_c = Dot(axes=2)([attention_softmax, image_encoderTrans])
decoder_2 = Add()([decoder_1, decoder_c])
decoder_softmax = Dense(21, activation='softmax')(decoder_2) 
#model
my_model = Model(inputs=[image_input, language_input, padding_input], outputs=decoder_softmax)
my_model.load_weights("/rap_blues/lunwen/convs2s/358_complete/org-weights-epoch-200--val_loss-0.3155--loss-0.2762.hdf5")

#prediction
actual, predicted = list(), list()
for i in range(len(texts)):
    print(i)
    yhat = generate_desc(my_model, tokenizer, train_features[i], max_length)
    print('\n\nReal---->\n\n' + texts[i])
    actual.append([texts[i].split()])
    predicted.append(yhat.split())


#save model
for i in range(len(predicted)):
    str=""
    for j in range(len(predicted[i])):
        str=str+predicted[i][j]+" "
    str=str[:-1]
    str=str+"\n"
    save_to_file("/rap_blues/lunwen/convs2s/kai/layer16/predicted.txt", str)













