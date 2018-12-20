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
from keras.initializers import RandomNormal

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


img_dim = 64
z_dim = 128
max_length = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 64
iters_per_sample = 10
total_iter = 10000
batch_size = 32
k = 2
p = 6

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

train_name = "/rap_blues/lunwen/generation/train/"
test_name = "/rap_blues/lunwen/generation/eval/"
xt, yt = load_data(test_name)
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

	
y_train = np.array(y_train)
y_test = np.array(y_test)





# 判别器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in
cond_in = Input(shape=(max_length, ))
x = Conv2D(img_dim,
           (5, 5),
           strides=(2, 2),
           padding='same')(x)
x = LeakyReLU()(x)
x = Conv2D(img_dim * 2,(5, 5),strides=(2, 2),padding='same')(x)
c = RepeatVector(256)(cond_in)
c = Reshape((16,16,128))(c)
x = concatenate([x, c])
x = BatchNormalization()(x)
x = LeakyReLU()(x)
for i in range(2):
    x = Conv2D(img_dim * 2**(i + 2),
               (5, 5),
               strides=(2, 2),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

x = Flatten()(x)
x = Dense(1, activation='sigmoid')(x)

d_model = Model([x_in,cond_in], x)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
cond_in = Input(shape=(max_length, ))
z = concatenate([z_in, cond_in])

z = Dense(4 * 4 * img_dim * 8)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((4, 4, img_dim * 8))(z)

for i in range(3):
    z = Conv2DTranspose(img_dim * 4 // 2**i,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)

g_model = Model([z_in,cond_in], z)
g_model.summary()


####################################################





# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
cond_in = Input(shape=(max_length, ))
g_model.trainable = False

x_real = x_in
x_fake = g_model([z_in, cond_in])

x_real_score = d_model([x_real, cond_in])
x_fake_score = d_model([x_fake, cond_in])

d_train_model = Model([x_in, z_in, cond_in],
                      [x_real_score, x_fake_score])



d_loss = K.mean(x_real_score - x_fake_score)

real_grad = K.gradients(x_real_score, [x_real])[0]
fake_grad = K.gradients(x_fake_score, [x_fake])[0]

real_grad_norm = K.sum(real_grad**2, axis=[1, 2, 3])**(p / 2)
fake_grad_norm = K.sum(fake_grad**2, axis=[1, 2, 3])**(p / 2)
grad_loss = K.mean(real_grad_norm + fake_grad_norm) * k / 2

w_dist = K.mean(x_fake_score - x_real_score)

d_train_model.add_loss(d_loss + grad_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))
d_train_model.metrics_names.append('w_dist')
d_train_model.metrics_tensors.append(w_dist)


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False

x_fake = g_model([z_in, cond_in])
x_fake_score = d_model([x_fake, cond_in])

g_train_model = Model([z_in, cond_in], x_fake_score)

g_loss = K.mean(x_fake_score)
g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))
g_train_model.load_weights("/rap_blues/MNIST/cwgan_test/samples/g_train_0model.weights") #good!!!

# 检查模型结构
d_train_model.summary()
g_train_model.summary()


# 采样函数
def sample(path):
    n = 9
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            yy = y_test[0][np.newaxis,:]
            z_sample = np.random.randn(1, z_dim)
            x_sample = g_model.predict([z_sample,yy])
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)


for i in range(total_iter):
    print(i)
    for j in range(1):
        z_sample = np.random.randn(1500, z_dim)
        d_train_model.fit([x_train, z_sample, y_train],None,  batch_size=batch_size, shuffle=False)
    for j in range(2):
        z_sample = np.random.randn(1500, z_dim)
        g_train_model.fit([z_sample, y_train],None, batch_size=batch_size, shuffle=False)
    if i % iters_per_sample == 0:
        sample('/rap_blues/MNIST/cwgan_test/samples/test_%s.png' % i)
        g_train_model.save_weights('/rap_blues/MNIST/cwgan_test/samples/g_train_%smodel.weights' % i)