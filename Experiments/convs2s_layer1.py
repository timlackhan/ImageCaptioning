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


def my_loadData(text_train_dir, image_train_dir, text_val_dir, image_val_dir):
    text = []
    images = []
    for i in range(254):
        readPath = image_train_dir + str(i+1) + ".png"
        img = np.array(Image.open(readPath))
        imgInfo = img.shape
        dstHeight = int(imgInfo[0]*0.25)
        dstWidth = int(imgInfo[1]*0.25)
        dst = cv2.resize(img,(dstWidth,dstHeight))/255.0
        images.append(dst)
    for i in range(104):
        readPath = image_val_dir + str(i+1) + ".png"
        img = np.array(Image.open(readPath))
        imgInfo = img.shape
        dstHeight = int(imgInfo[0]*0.25)
        dstWidth = int(imgInfo[1]*0.25)
        dst = cv2.resize(img,(dstWidth,dstHeight))/255.0
        images.append(dst)
    images = np.array(images, dtype=float)
    text_train = open(text_train_dir,"r")
    for i in range(254):
        line = text_train.readline().strip("\n")
        text.append(line)
    text_train.close()
    text_val = open(text_val_dir,"r")
    for i in range(104):
        line = text_val.readline().strip("\n")
        text.append(line)
    text_val.close()
    return images, text
	
image_train_dir = "/rap_blues/train_imgs/"
image_val_dir = "/rap_blues/val_imgs/"
text_train_dir = "/rap_blues/train_labels2/train_labels2.txt"
text_val_dir = "/rap_blues/val_labels2/val_labels2.txt"
train_features, texts = my_loadData(text_train_dir, image_train_dir, text_val_dir, image_val_dir)



tokenizer = Tokenizer(filters='', split=" ", lower=False)
tokenizer.fit_on_texts([load_doc('/rap_blues/my_bootstrap.vocab')])
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(texts)
max_sequence = max(len(s) for s in train_sequences)
max_length = 48


#in_seq:  48 dim  [0,0,0,17]
#out_seq: 48 dim  [0,0,17,18]
def preprocess_data(sequences, features):
    X, y, padding_upper, image_data = list(), list(), list(), list()
    for img_no, seq in enumerate(sequences):
        for i in range(1, len(seq)):
            # Add the sentence until the current count(i) and add the current count to the output
            in_seq, out_seq = seq[:i], seq[:(i+1)]
            # Pad all the input token sentences to max_sequence
            in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
            # Turn the output into one-hot encoding
            out_seq = pad_sequences([out_seq], maxlen=max_sequence)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # Add the corresponding image to the boostrap token file
            image_data.append(features[img_no])
            # Cap the input sentence to 48 tokens and add it
            X.append(in_seq[-48:])
            y.append(out_seq[-48:])
            padding_upper.append(np.zeros((2, 21)))
    return np.array(X), np.array(y), np.array(image_data), np.array(padding_upper)
  

    
X, y, image_data, padding_up = preprocess_data(train_sequences, train_features)


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
# 1st block
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
my_model.load_weights("/rap_blues/lunwen/convs2s/220/org-weights-epoch-64--val_loss-0.3232--loss-0.3156.hdf5")


json_file = open('/rap_blues/sketch_code/model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/rap_blues/sketch_code/weights.h5")
my_model.layers[2].set_weights(model.layers[4].layers[0].get_weights())
my_model.layers[5].set_weights(model.layers[4].layers[1].get_weights())
my_model.layers[7].set_weights(model.layers[4].layers[2].get_weights())
my_model.layers[10].set_weights(model.layers[4].layers[3].get_weights())


#train
optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
my_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
filepath="/rap_blues/lunwen/convs2s/358_complete/org-weights-epoch-{epoch:02d}--val_loss-{val_loss:.4f}--loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=4)
reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0, mode='min', epsilon=0.0001, cooldown=0, min_lr=0)  
callbacks_list = [checkpoint, reduceLR]
my_model.fit([image_data, X, padding_up], y, batch_size=128, shuffle=False, validation_split=0.1, callbacks=callbacks_list, verbose=1, epochs=500)








