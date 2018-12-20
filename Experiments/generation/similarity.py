from keras.layers import MaxPool2D, GlobalAveragePooling2D, Subtract, BatchNormalization, LeakyReLU, Conv2DTranspose, Multiply,Permute,merge,Bidirectional,add, Embedding, TimeDistributed, RepeatVector, LSTM, GRU, concatenate , Input, Reshape, Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
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
import keras.backend.tensorflow_backend as KTF
import keras.backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from os import listdir
from scipy import misc
from keras.callbacks import Callback
import imageio
import glob


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


img_dim = 64
z_dim = 128
max_length = 128


def load_doc(filename):
    file = open(filename, 'r', encoding='ISO-8859-15')
    text = file.read()
    file.close()
    return text

   
def load_data(data_dir):
    text = []
    images = []
    # Load all the files and order them
    all_filenames = listdir(data_dir)
    all_filenames.sort()
    for filename in (all_filenames):
        if filename[-3:] == "npz":
            # Load the images already prepared in arrays
            image = np.load(data_dir+filename)
            image = misc.imresize(image['features'], (64, 64))
            images.append(image)
        else:
            # Load the boostrap tokens and rap them in a start and end tag
            syntax = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            # Seperate all the words with a single space
            syntax = ' '.join(syntax.split())
            # Add a space after each comma
            syntax = syntax.replace(',', ' ,')
            text.append(syntax)
    images = np.array(images, dtype=float)
    return images, text

# load datasets
train_name = "/rap_blues/lunwen/generation/train/"
test_name = "/rap_blues/lunwen/generation/eval/"
xt, yt = load_data(test_name)



def get_mse(model):
    n = 250
    mse = 0
    figure = np.zeros((img_dim, img_dim, 3))
    for i in range(n):
        yy = y_test[i][np.newaxis,:]
        z_sample = np.random.randn(1, z_dim)
        x_sample = model.predict([z_sample,yy])
        digit = x_sample[0]
        figure = digit
        figure = (figure + 1) / 2 * 255
        figure = np.round(figure, 0).astype(int)
        mse = mse + np.mean((figure - xt[i])**2)
    return mse/250.0
	
	
def get_similarity(model):
    n = 250
    sim = 0
    figure = np.zeros((img_dim, img_dim, 3))
    for i in range(n):
        yy = y_test[i][np.newaxis,:]
        z_sample = np.random.randn(1, z_dim)
        x_sample = model.predict([z_sample,yy])
        digit = x_sample[0]
        figure = digit
        figure = (figure + 1) / 2 * 255
        figure = np.round(figure, 0).astype(int)
        num = float(np.matmul(figure.reshape((1, img_dim * img_dim * 3)),xt[i].reshape((img_dim * img_dim * 3, 1))))
        denorm_A = np.linalg.norm(figure.reshape((1, img_dim * img_dim * 3)))
        denorm_B = np.linalg.norm(xt[i].reshape((img_dim * img_dim * 3, 1)))
        sim_temp = num / (denorm_A*denorm_B)
        sim = 0.5 + 0.5 * sim_temp + sim
    return sim/250.0


