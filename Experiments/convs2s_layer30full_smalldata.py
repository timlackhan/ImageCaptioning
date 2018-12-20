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
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


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
train_features = train_features[30:50]
texts = texts[30:50]


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
image_conv3 = Conv2D(16, (3,3), activation='relu', padding='same')(image_conv1)
image_conv4 = Conv2D(128, (3,3), activation='relu', padding='same', strides=8)(image_conv3)
image_conv5 = Conv2D(21, (3,3), activation='relu', padding='same')(image_conv4)
image_encoder = Reshape((1024,21))(image_conv5)
image_encoderTrans = Reshape((21,1024))(image_encoder)
#decoder 
language_input = Input(shape=(max_length,))
padding_input = Input(shape=(2, 21, ))
language_model = Embedding(vocab_size, 21, input_length=max_length, mask_zero=False)(language_input)

# 1st block
padding_language = concatenate([padding_input, language_model], axis=1)
decoder_conv1 = Conv1D(21, 3, padding='valid')(padding_language)
decoder_gate1 = Conv1D(21, 3, padding='valid', activation='sigmoid')(padding_language)
decoder_glu1 =  Multiply()([decoder_conv1, decoder_gate1]) 
decoder_1 = Add()([language_model, decoder_glu1])
# 1st attention
attention_matrix1 = Dot(axes=2)([decoder_1, image_encoder])
attention_softmax1 = Activation('softmax')(attention_matrix1)
decoder_c1 = Dot(axes=2)([attention_softmax1, image_encoderTrans])
decoder_2i = Add()([decoder_1, decoder_c1])
# 2nd block
decoder_2o = concatenate([padding_input, decoder_2i], axis=1)
decoder_conv2 = Conv1D(21, 3, padding='valid')(decoder_2o)
decoder_gate2 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_2o)
decoder_glu2 =  Multiply()([decoder_conv2, decoder_gate2]) 
decoder_2 = Add()([decoder_2i, decoder_glu2])
# 2nd attention
attention_matrix2 = Dot(axes=2)([decoder_2, image_encoder])
attention_softmax2 = Activation('softmax')(attention_matrix2)
decoder_c2 = Dot(axes=2)([attention_softmax2, image_encoderTrans])
decoder_3i = Add()([decoder_2, decoder_c2])
# 3rd block
decoder_3o = concatenate([padding_input, decoder_3i], axis=1)
decoder_conv3 = Conv1D(21, 3, padding='valid')(decoder_3o)
decoder_gate3 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_3o)
decoder_glu3 =  Multiply()([decoder_conv3, decoder_gate3]) 
decoder_3 = Add()([decoder_3i, decoder_glu3])
# 3rd attention
attention_matrix3 = Dot(axes=2)([decoder_3, image_encoder])
attention_softmax3 = Activation('softmax')(attention_matrix3)
decoder_c3 = Dot(axes=2)([attention_softmax3, image_encoderTrans])
decoder_4i = Add()([decoder_3, decoder_c3])
# 4th block
decoder_4o = concatenate([padding_input, decoder_4i], axis=1)
decoder_conv4 = Conv1D(21, 3, padding='valid')(decoder_4o)
decoder_gate4 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_4o)
decoder_glu4 =  Multiply()([decoder_conv4, decoder_gate4]) 
decoder_4 = Add()([decoder_4i, decoder_glu4])
# 4th attention
attention_matrix4 = Dot(axes=2)([decoder_4, image_encoder])
attention_softmax4 = Activation('softmax')(attention_matrix4)
decoder_c4 = Dot(axes=2)([attention_softmax4, image_encoderTrans])
decoder_5i = Add()([decoder_4, decoder_c4])
# 5th block
decoder_5o = concatenate([padding_input, decoder_5i], axis=1)
decoder_conv5 = Conv1D(21, 3, padding='valid')(decoder_5o)
decoder_gate5 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_5o)
decoder_glu5 =  Multiply()([decoder_conv5, decoder_gate5]) 
decoder_5 = Add()([decoder_5i, decoder_glu5])
# 5th attention
attention_matrix5 = Dot(axes=2)([decoder_5, image_encoder])
attention_softmax5 = Activation('softmax')(attention_matrix5)
decoder_c5 = Dot(axes=2)([attention_softmax5, image_encoderTrans])
decoder_6i = Add()([decoder_5, decoder_c5])
# 6th block
decoder_6o = concatenate([padding_input, decoder_6i], axis=1)
decoder_conv6 = Conv1D(21, 3, padding='valid')(decoder_6o)
decoder_gate6 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_6o)
decoder_glu6 =  Multiply()([decoder_conv6, decoder_gate6]) 
decoder_6 = Add()([decoder_6i, decoder_glu6])
# 6th attention
attention_matrix6 = Dot(axes=2)([decoder_6, image_encoder])
attention_softmax6 = Activation('softmax')(attention_matrix6)
decoder_c6 = Dot(axes=2)([attention_softmax6, image_encoderTrans])
decoder_7i = Add()([decoder_6, decoder_c6])
# 7th block
decoder_7o = concatenate([padding_input, decoder_7i], axis=1)
decoder_conv7 = Conv1D(21, 3, padding='valid')(decoder_7o)
decoder_gate7 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_7o)
decoder_glu7 =  Multiply()([decoder_conv7, decoder_gate7]) 
decoder_7 = Add()([decoder_7i, decoder_glu7])
# 7th attention
attention_matrix7 = Dot(axes=2)([decoder_7, image_encoder])
attention_softmax7 = Activation('softmax')(attention_matrix7)
decoder_c7 = Dot(axes=2)([attention_softmax7, image_encoderTrans])
decoder_8i = Add()([decoder_7, decoder_c7])
# 8th block
decoder_8o = concatenate([padding_input, decoder_8i], axis=1)
decoder_conv8 = Conv1D(21, 3, padding='valid')(decoder_8o)
decoder_gate8 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_8o)
decoder_glu8 =  Multiply()([decoder_conv8, decoder_gate8]) 
decoder_8 = Add()([decoder_8i, decoder_glu8])
# 8th attention
attention_matrix8 = Dot(axes=2)([decoder_8, image_encoder])
attention_softmax8 = Activation('softmax')(attention_matrix8)
decoder_c8 = Dot(axes=2)([attention_softmax8, image_encoderTrans])
decoder_9i = Add()([decoder_8, decoder_c8])
# 9th block
decoder_9o = concatenate([padding_input, decoder_9i], axis=1)
decoder_conv9 = Conv1D(21, 3, padding='valid')(decoder_9o)
decoder_gate9 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_9o)
decoder_glu9 =  Multiply()([decoder_conv9, decoder_gate9]) 
decoder_9 = Add()([decoder_9i, decoder_glu9])
# 9th attention
attention_matrix9 = Dot(axes=2)([decoder_9, image_encoder])
attention_softmax9 = Activation('softmax')(attention_matrix9)
decoder_c9 = Dot(axes=2)([attention_softmax9, image_encoderTrans])
decoder_10i = Add()([decoder_9, decoder_c9])
# 10th block
decoder_10o = concatenate([padding_input, decoder_10i], axis=1)
decoder_conv10 = Conv1D(21, 3, padding='valid')(decoder_10o)
decoder_gate10 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_10o)
decoder_glu10 =  Multiply()([decoder_conv10, decoder_gate10]) 
decoder_10 = Add()([decoder_10i, decoder_glu10])
# 10th attention
attention_matrix10 = Dot(axes=2)([decoder_10, image_encoder])
attention_softmax10 = Activation('softmax')(attention_matrix10)
decoder_c10 = Dot(axes=2)([attention_softmax10, image_encoderTrans])
decoder_11i = Add()([decoder_10, decoder_c10])
# 11th block
decoder_11o = concatenate([padding_input, decoder_11i], axis=1)
decoder_conv11 = Conv1D(21, 3, padding='valid')(decoder_11o)
decoder_gate11 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_11o)
decoder_glu11 =  Multiply()([decoder_conv11, decoder_gate11]) 
decoder_11 = Add()([decoder_11i, decoder_glu11])
# 11th attention
attention_matrix11 = Dot(axes=2)([decoder_11, image_encoder])
attention_softmax11 = Activation('softmax')(attention_matrix11)
decoder_c11 = Dot(axes=2)([attention_softmax11, image_encoderTrans])
decoder_12i = Add()([decoder_11, decoder_c11])
# 12th block
decoder_12o = concatenate([padding_input, decoder_12i], axis=1)
decoder_conv12 = Conv1D(21, 3, padding='valid')(decoder_12o)
decoder_gate12 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_12o)
decoder_glu12 =  Multiply()([decoder_conv12, decoder_gate12]) 
decoder_12 = Add()([decoder_12i, decoder_glu12])
# 12th attention
attention_matrix12 = Dot(axes=2)([decoder_12, image_encoder])
attention_softmax12 = Activation('softmax')(attention_matrix12)
decoder_c12 = Dot(axes=2)([attention_softmax12, image_encoderTrans])
decoder_13i = Add()([decoder_12, decoder_c12])
# 13th block
decoder_13o = concatenate([padding_input, decoder_13i], axis=1)
decoder_conv13 = Conv1D(21, 3, padding='valid')(decoder_13o)
decoder_gate13 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_13o)
decoder_glu13 =  Multiply()([decoder_conv13, decoder_gate13]) 
decoder_13 = Add()([decoder_13i, decoder_glu13])
# 13th attention
attention_matrix13 = Dot(axes=2)([decoder_13, image_encoder])
attention_softmax13 = Activation('softmax')(attention_matrix13)
decoder_c13 = Dot(axes=2)([attention_softmax13, image_encoderTrans])
decoder_14i = Add()([decoder_13, decoder_c13])
# 14th block
decoder_14o = concatenate([padding_input, decoder_14i], axis=1)
decoder_conv14 = Conv1D(21, 3, padding='valid')(decoder_14o)
decoder_gate14 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_14o)
decoder_glu14 =  Multiply()([decoder_conv14, decoder_gate14]) 
decoder_14 = Add()([decoder_14i, decoder_glu14])
# 14th attention
attention_matrix14 = Dot(axes=2)([decoder_14, image_encoder])
attention_softmax14 = Activation('softmax')(attention_matrix14)
decoder_c14 = Dot(axes=2)([attention_softmax14, image_encoderTrans])
decoder_15i = Add()([decoder_14, decoder_c14])
# 15th block
decoder_15o = concatenate([padding_input, decoder_15i], axis=1)
decoder_conv15 = Conv1D(21, 3, padding='valid')(decoder_15o)
decoder_gate15 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_15o)
decoder_glu15 =  Multiply()([decoder_conv15, decoder_gate15]) 
decoder_15 = Add()([decoder_15i, decoder_glu15])
# 15th attention
attention_matrix15 = Dot(axes=2)([decoder_15, image_encoder])
attention_softmax15 = Activation('softmax')(attention_matrix15)
decoder_c15 = Dot(axes=2)([attention_softmax15, image_encoderTrans])
decoder_16i = Add()([decoder_15, decoder_c15])
# 16th block
decoder_16o = concatenate([padding_input, decoder_16i], axis=1)
decoder_conv16 = Conv1D(21, 3, padding='valid')(decoder_16o)
decoder_gate16 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_16o)
decoder_glu16 =  Multiply()([decoder_conv16, decoder_gate16]) 
decoder_16 = Add()([decoder_16i, decoder_glu16])
# 16th attention
attention_matrix16 = Dot(axes=2)([decoder_16, image_encoder])
attention_softmax16 = Activation('softmax')(attention_matrix16)
decoder_c16 = Dot(axes=2)([attention_softmax16, image_encoderTrans])
decoder_17i = Add()([decoder_16, decoder_c16])
# 17th block
decoder_17o = concatenate([padding_input, decoder_17i], axis=1)
decoder_conv17 = Conv1D(21, 3, padding='valid')(decoder_17o)
decoder_gate17 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_17o)
decoder_glu17 =  Multiply()([decoder_conv17, decoder_gate17]) 
decoder_17 = Add()([decoder_17i, decoder_glu17])
# 17th attention
attention_matrix17 = Dot(axes=2)([decoder_17, image_encoder])
attention_softmax17 = Activation('softmax')(attention_matrix17)
decoder_c17 = Dot(axes=2)([attention_softmax17, image_encoderTrans])
decoder_18i = Add()([decoder_17, decoder_c17])
# 18th block
decoder_18o = concatenate([padding_input, decoder_18i], axis=1)
decoder_conv18 = Conv1D(21, 3, padding='valid')(decoder_18o)
decoder_gate18 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_18o)
decoder_glu18 =  Multiply()([decoder_conv18, decoder_gate18]) 
decoder_18 = Add()([decoder_18i, decoder_glu18])
# 18th attention
attention_matrix18 = Dot(axes=2)([decoder_18, image_encoder])
attention_softmax18 = Activation('softmax')(attention_matrix18)
decoder_c18 = Dot(axes=2)([attention_softmax18, image_encoderTrans])
decoder_19i = Add()([decoder_18, decoder_c18])
# 19th block
decoder_19o = concatenate([padding_input, decoder_19i], axis=1)
decoder_conv19 = Conv1D(21, 3, padding='valid')(decoder_19o)
decoder_gate19 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_19o)
decoder_glu19 =  Multiply()([decoder_conv19, decoder_gate19]) 
decoder_19 = Add()([decoder_19i, decoder_glu19])
# 19th attention
attention_matrix19 = Dot(axes=2)([decoder_19, image_encoder])
attention_softmax19 = Activation('softmax')(attention_matrix19)
decoder_c19 = Dot(axes=2)([attention_softmax19, image_encoderTrans])
decoder_20i = Add()([decoder_19, decoder_c19])
# 20th block
decoder_20o = concatenate([padding_input, decoder_20i], axis=1)
decoder_conv20 = Conv1D(21, 3, padding='valid')(decoder_20o)
decoder_gate20 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_20o)
decoder_glu20 =  Multiply()([decoder_conv20, decoder_gate20]) 
decoder_20 = Add()([decoder_20i, decoder_glu20])
# 20th attention
attention_matrix20 = Dot(axes=2)([decoder_20, image_encoder])
attention_softmax20 = Activation('softmax')(attention_matrix20)
decoder_c20 = Dot(axes=2)([attention_softmax20, image_encoderTrans])
decoder_21i = Add()([decoder_20, decoder_c20])
# 21th block
decoder_21o = concatenate([padding_input, decoder_21i], axis=1)
decoder_conv21 = Conv1D(21, 3, padding='valid')(decoder_21o)
decoder_gate21 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_21o)
decoder_glu21 =  Multiply()([decoder_conv21, decoder_gate21]) 
decoder_21 = Add()([decoder_21i, decoder_glu21])
# 21th attention
attention_matrix21 = Dot(axes=2)([decoder_21, image_encoder])
attention_softmax21 = Activation('softmax')(attention_matrix21)
decoder_c21 = Dot(axes=2)([attention_softmax21, image_encoderTrans])
decoder_22i = Add()([decoder_21, decoder_c21])
# 22th block
decoder_22o = concatenate([padding_input, decoder_22i], axis=1)
decoder_conv22 = Conv1D(21, 3, padding='valid')(decoder_22o)
decoder_gate22 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_22o)
decoder_glu22 =  Multiply()([decoder_conv22, decoder_gate22]) 
decoder_22 = Add()([decoder_22i, decoder_glu22])
# 22th attention
attention_matrix22 = Dot(axes=2)([decoder_22, image_encoder])
attention_softmax22 = Activation('softmax')(attention_matrix22)
decoder_c22 = Dot(axes=2)([attention_softmax22, image_encoderTrans])
decoder_23i = Add()([decoder_22, decoder_c22])
# 23th block
decoder_23o = concatenate([padding_input, decoder_23i], axis=1)
decoder_conv23 = Conv1D(21, 3, padding='valid')(decoder_23o)
decoder_gate23 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_23o)
decoder_glu23 =  Multiply()([decoder_conv23, decoder_gate23]) 
decoder_23 = Add()([decoder_23i, decoder_glu23])
# 23th attention
attention_matrix23 = Dot(axes=2)([decoder_23, image_encoder])
attention_softmax23 = Activation('softmax')(attention_matrix23)
decoder_c23 = Dot(axes=2)([attention_softmax23, image_encoderTrans])
decoder_24i = Add()([decoder_23, decoder_c23])
# 24th block
decoder_24o = concatenate([padding_input, decoder_24i], axis=1)
decoder_conv24 = Conv1D(21, 3, padding='valid')(decoder_24o)
decoder_gate24 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_24o)
decoder_glu24 =  Multiply()([decoder_conv24, decoder_gate24]) 
decoder_24 = Add()([decoder_24i, decoder_glu24])
# 24th attention
attention_matrix24 = Dot(axes=2)([decoder_24, image_encoder])
attention_softmax24 = Activation('softmax')(attention_matrix24)
decoder_c24 = Dot(axes=2)([attention_softmax24, image_encoderTrans])
decoder_25i = Add()([decoder_24, decoder_c24])
# 25th block
decoder_25o = concatenate([padding_input, decoder_25i], axis=1)
decoder_conv25 = Conv1D(21, 3, padding='valid')(decoder_25o)
decoder_gate25 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_25o)
decoder_glu25 =  Multiply()([decoder_conv25, decoder_gate25]) 
decoder_25 = Add()([decoder_25i, decoder_glu25])
# 25th attention
attention_matrix25 = Dot(axes=2)([decoder_25, image_encoder])
attention_softmax25 = Activation('softmax')(attention_matrix25)
decoder_c25 = Dot(axes=2)([attention_softmax25, image_encoderTrans])
decoder_26i = Add()([decoder_25, decoder_c25])
# 26th block
decoder_26o = concatenate([padding_input, decoder_26i], axis=1)
decoder_conv26 = Conv1D(21, 3, padding='valid')(decoder_26o)
decoder_gate26 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_26o)
decoder_glu26 =  Multiply()([decoder_conv26, decoder_gate26]) 
decoder_26 = Add()([decoder_26i, decoder_glu26])
# 26th attention
attention_matrix26 = Dot(axes=2)([decoder_26, image_encoder])
attention_softmax26 = Activation('softmax')(attention_matrix26)
decoder_c26 = Dot(axes=2)([attention_softmax26, image_encoderTrans])
decoder_27i = Add()([decoder_26, decoder_c26])
# 27th block
decoder_27o = concatenate([padding_input, decoder_27i], axis=1)
decoder_conv27 = Conv1D(21, 3, padding='valid')(decoder_27o)
decoder_gate27 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_27o)
decoder_glu27 =  Multiply()([decoder_conv27, decoder_gate27]) 
decoder_27 = Add()([decoder_27i, decoder_glu27])
# 27th attention
attention_matrix27 = Dot(axes=2)([decoder_27, image_encoder])
attention_softmax27 = Activation('softmax')(attention_matrix27)
decoder_c27 = Dot(axes=2)([attention_softmax27, image_encoderTrans])
decoder_28i = Add()([decoder_27, decoder_c27])
# 28th block
decoder_28o = concatenate([padding_input, decoder_28i], axis=1)
decoder_conv28 = Conv1D(21, 3, padding='valid')(decoder_28o)
decoder_gate28 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_28o)
decoder_glu28 =  Multiply()([decoder_conv28, decoder_gate28]) 
decoder_28 = Add()([decoder_28i, decoder_glu28])
# 28th attention
attention_matrix28 = Dot(axes=2)([decoder_28, image_encoder])
attention_softmax28 = Activation('softmax')(attention_matrix28)
decoder_c28 = Dot(axes=2)([attention_softmax28, image_encoderTrans])
decoder_29i = Add()([decoder_28, decoder_c28])
# 29th block
decoder_29o = concatenate([padding_input, decoder_29i], axis=1)
decoder_conv29 = Conv1D(21, 3, padding='valid')(decoder_29o)
decoder_gate29 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_29o)
decoder_glu29 =  Multiply()([decoder_conv29, decoder_gate29]) 
decoder_29 = Add()([decoder_29i, decoder_glu29])
# 29th attention
attention_matrix29 = Dot(axes=2)([decoder_29, image_encoder])
attention_softmax29 = Activation('softmax')(attention_matrix29)
decoder_c29 = Dot(axes=2)([attention_softmax29, image_encoderTrans])
decoder_30i = Add()([decoder_29, decoder_c29])
# 30th block
decoder_30o = concatenate([padding_input, decoder_30i], axis=1)
decoder_conv30 = Conv1D(21, 3, padding='valid')(decoder_30o)
decoder_gate30 = Conv1D(21, 3, padding='valid', activation='sigmoid')(decoder_30o)
decoder_glu30 =  Multiply()([decoder_conv30, decoder_gate30]) 
decoder_30 = Add()([decoder_30i, decoder_glu30])
# 30th attention
attention_matrix30 = Dot(axes=2)([decoder_30, image_encoder])
attention_softmax30 = Activation('softmax')(attention_matrix30)
decoder_c30 = Dot(axes=2)([attention_softmax30, image_encoderTrans])
decoder_31i = Add()([decoder_30, decoder_c30])


decoder_softmax = Dense(21, activation='softmax')(decoder_31i) 
#model
my_model = Model(inputs=[image_input, language_input, padding_input], outputs=decoder_softmax)
my_model.load_weights("/rap_blues/lunwen/convs2s/layer_30_alldata/org-weights-epoch-460--val_loss-1.0461--loss-0.1066.hdf5")


json_file = open('/rap_blues/sketch_code/model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/rap_blues/sketch_code/weights.h5")
my_model.layers[4].set_weights(model.layers[4].layers[0].get_weights())
my_model.layers[6].set_weights(model.layers[4].layers[1].get_weights())
my_model.layers[6].set_weights(model.layers[4].layers[2].get_weights())
my_model.layers[8].set_weights(model.layers[4].layers[3].get_weights())


#train
optimizer = Adam(lr=0.0001, clipvalue=1.0)
my_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
filepath="/rap_blues/lunwen/convs2s/layer_30full_smalldata/org-weights-epoch-{epoch:02d}--val_loss-{val_loss:.4f}--loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=30)
reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0, mode='min', epsilon=0.0001, cooldown=0, min_lr=0)  
callbacks_list = [checkpoint, reduceLR]
my_model.fit([image_data, X, padding_up], y, batch_size=128, shuffle=False, validation_split=0.1, callbacks=callbacks_list, verbose=1, epochs=2000)










my_model.layers[4].layers[0].set_weights(model.layers[4].layers[0].get_weights())
my_model.layers[4].layers[1].set_weights(model.layers[4].layers[1].get_weights())
my_model.layers[4].layers[4].set_weights(model.layers[4].layers[8].get_weights())
my_model.layers[4].layers[6].set_weights(model.layers[4].layers[10].get_weights())








