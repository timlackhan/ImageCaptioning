from keras.layers import GlobalAveragePooling2D, Subtract, BatchNormalization, LeakyReLU, Conv2DTranspose, Multiply,Permute,merge,Bidirectional,add, Embedding, TimeDistributed, RepeatVector, LSTM, GRU, concatenate , Input, Reshape, Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


img_dim = 64
z_dim = 512
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
x_train, y_train = load_data(train_name)
x_test, y_test = load_data(test_name)
x_train = x_train.astype('float32') / 255.* 2 - 1
x_test = x_test.astype('float32') / 255.* 2 - 1
tokenizer = Tokenizer(filters='', split=" ", lower=False)
tokenizer.fit_on_texts([load_doc('/rap_blues/bootstrap.vocab')])
vocab_size = len(tokenizer.word_index) + 1
y_train = tokenizer.texts_to_sequences(y_train)
y_test = tokenizer.texts_to_sequences(y_test)
for i in range(len(y_train)):
	y_train[i] = pad_sequences([y_train[i]], maxlen=max_length)[0]
for i in range(len(y_test)):
	y_test[i] = pad_sequences([y_test[i]], maxlen=max_length)[0]
x_train = x_train.reshape((1500, original_dim))
x_test = x_test.reshape((250, original_dim))
y_train = np.array(y_train)
y_test = np.array(y_test)


def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(x_train)
        for f in x_train:
            X.append(f)
            if len(X) == batch_size:
                X = np.array(X)
                yield X,None
                X = []

x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in
x = Conv2D(int(z_dim/16), kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(int(z_dim/8), kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(int(z_dim/4), kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(int(z_dim/2), kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(z_dim, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = GlobalAveragePooling2D()(x)
encoder = Model(x_in, x)
encoder.summary()
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]



z_in = Input(shape=K.int_shape(x)[1:])
z = z_in
z = Dense(np.prod(map_size)*z_dim)(z)
z = Reshape(map_size + (z_dim,))(z)
z = Conv2DTranspose(int(z_dim/2), kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(int(z_dim/4), kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(int(z_dim/8), kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(int(z_dim/16), kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(3, kernel_size=(5,5), strides=(2,2), padding='SAME')(z)
z = Activation('tanh')(z)


decoder = Model(z_in, z)
decoder.summary()

class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        z, shift, log_scale = inputs
        z = K.exp(log_scale) * z + shift
        logdet = -K.sum(K.mean(log_scale, 0))
        self.add_loss(logdet)
        return z

z_shift = Dense(z_dim)(x)
z_log_scale = Dense(z_dim)(x)
u = Lambda(lambda z: K.random_normal(shape=K.shape(z)))(z_shift)
z = ScaleShift()([u, z_shift, z_log_scale])

x_recon = decoder(z)
x_out = Subtract()([x_in, x_recon])

recon_loss = 0.5 * K.sum(K.mean(x_out**2, 0)) + 0.5 * np.log(2*np.pi) * np.prod(K.int_shape(x_out)[1:])
z_loss = 0.5 * K.sum(K.mean(z**2, 0)) - 0.5 * K.sum(K.mean(u**2, 0))
vae_loss = recon_loss + z_loss

vae = Model(x_in, x_out)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(1e-4))


def sample(path):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    for i in range(n):
        for j in range(n):
            x_recon = decoder.predict(np.random.randn(1, *K.int_shape(x)[1:]))
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    imageio.imwrite(path, figure)


class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('/rap_blues/MNIST/cvae_images/samples'):
            os.mkdir('/rap_blues/MNIST/cvae_images/samples')
    def on_epoch_end(self, epoch, logs=None):
        path = '/rap_blues/MNIST/cvae_images/samples/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('./best_encoder.weights')


evaluator = Evaluate()

vae.fit_generator(data_generator(),
                  epochs=1000,
                  steps_per_epoch=1000,
                  callbacks=[evaluator])