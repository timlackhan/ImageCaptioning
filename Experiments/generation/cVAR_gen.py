from keras.layers import GlobalAveragePooling2D, BatchNormalization, LeakyReLU, Multiply,Permute,merge,Bidirectional,add, Embedding, TimeDistributed, RepeatVector, LSTM, GRU, concatenate , Input, Reshape, Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
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
from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import mnist
from scipy.stats import norm
from keras import metrics
from os import listdir
from scipy import misc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)



batch_size = 100
original_dim = 12288
latent_dim = 20 # in order to draw
intermediate_dim = 256
epochs = 1000
epsilon_std = 1.0
max_length = 150

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
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
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


x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# mean and log_sigma2 of p(Z|X)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# input category
y = Input(shape=(max_length,))
 
yh = Dense(latent_dim)(y) # the mean of every category

# reparameterize trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# add noise to inputs
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder, also is generator
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# build model
vae = Model([x, y], [x_decoded_mean, yh])

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)

# 只需要修改K.square(z_mean)为K.square(z_mean - yh)，也就是让隐变量向类内均值看齐
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean - yh) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


filepath="/rap_blues/lunwen/generation/cvae/org-weights-epoch-{epoch:02d}--val_loss-{val_loss:.4f}--loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=4)
reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0, mode='min', epsilon=0.0001, cooldown=0, min_lr=0)  
callbacks_list = [checkpoint, reduceLR]





vae.fit([x_train, y_train], 
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
		callbacks=callbacks_list,
        validation_data=([x_test, y_test], None))


# 构建encoder，然后观察各个数字在隐空间的分布
encoder = Model(x, z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.colorbar()
plt.show()

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# 输出每个类的均值向量
mu = Model(y, yh)
mu = mu.predict(np.eye(num_classes))

# 观察能否通过控制隐变量的均值来输出特定类别的数字
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

output_digit = 9 # 指定输出数字

# 用正态分布的分位数来构建隐变量对
grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) + mu[output_digit][1]
grid_y = norm.ppf(np.linspace(0.05, 0.95, n)) + mu[output_digit][0]

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
-----------------------------------------------------------------------------------------
pic_num = 0
variations = 30 # rate of change; higher is slower
for j in range(n_z, n_z + n_y - 1):
	for k in range(variations):
		v = np.zeros((1, n_z+n_y))
		v[0, j] = 1 - (k/variations)
		v[0, j+1] = (k/variations)
		generated = decoder.predict(v)
		pic_idx = j - n_z + (k/variations)
		file_name = '/rap_blues/MNIST/cvae_images/img{0:.3f}.jpg'.format(pic_idx)
		imsave(file_name, generated.reshape((28,28)))
		pic_num += 1
		
		
		
vae.fit([x_train, y_train], 
        shuffle=True,
        epochs=1000,
        batch_size=batch_size,
        validation_data=([x_test, y_test], None))
		
from scipy.misc import imsave		
file_name = '/rap_blues/MNIST/cvae_images/aa{0:.3f}.jpg'.format(-1)
imsave(file_name, figure)


