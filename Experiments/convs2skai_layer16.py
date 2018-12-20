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
train_features=train_features[220:]
texts=texts[220:]


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
            padding_upper.append(np.zeros((2, 48)))
    return np.array(X), np.array(y), np.array(image_data), np.array(padding_upper)

    
X, y, image_data, padding_up = preprocess_data(train_sequences, train_features)


#encoder kernel=3
image_input = Input(shape=(256, 256, 3,))
image_conv1 = Conv2D(16, (3,3), padding='valid', activation='relu', name='conv1')(image_input)
image_conv2 = Conv2D(16, (3,3), activation='relu', padding='same', strides=2, name='conv2')(image_conv1)
image_conv3 = Conv2D(32, (3,3), activation='relu', padding='same', name='conv3')(image_conv2)
image_conv4 = Conv2D(32, (3,3), activation='relu', padding='same', strides=2, name='conv4')(image_conv3)
image_conv5 = Conv2D(64, (3,3), activation='relu', padding='same', name='conv5')(image_conv4)
image_conv6 = Conv2D(64, (3,3), activation='relu', padding='same', strides=2, name='conv6')(image_conv5)
image_conv7 = Conv2D(128, (3,3), activation='relu', padding='same', name='conv7')(image_conv6)
image_conv8 = Flatten()(image_conv7)
image_conv9 = Dense(1024, activation='relu', name='conv9')(image_conv8)
image_conv10 = Dropout(0.3)(image_conv9)
image_conv11 = Dense(1024, activation='relu', name='conv11')(image_conv10)
image_conv12 = Dropout(0.3)(image_conv11)
image_encoderTrans = RepeatVector(max_length)(image_conv12)
image_encoder = Reshape((1024, 48))(image_encoderTrans)
#decoder 
language_input = Input(shape=(max_length,))
padding_input = Input(shape=(2, 48, ))
language_model = Embedding(vocab_size, 48, input_length=max_length, mask_zero=False)(language_input)

# 1st block
padding_language = concatenate([padding_input, language_model], axis=1)
decoder_conv1 = Conv1D(48, 3, padding='valid')(padding_language)
decoder_gate1 = Conv1D(48, 3, padding='valid', activation='sigmoid')(padding_language)
decoder_glu1 =  Multiply()([decoder_conv1, decoder_gate1]) 
decoder_1 = Add()([language_model, decoder_glu1])
# 1st attention
attention_matrix1 = Dot(axes=2)([decoder_1, image_encoder])
attention_softmax1 = Activation('softmax')(attention_matrix1)
decoder_c1 = Dot(axes=2)([attention_softmax1, image_encoderTrans])
decoder_2i = Add()([decoder_1, decoder_c1])
# 2nd block
decoder_2o = concatenate([padding_input, decoder_2i], axis=1)
decoder_conv2 = Conv1D(48, 3, padding='valid')(decoder_2o)
decoder_gate2 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_2o)
decoder_glu2 =  Multiply()([decoder_conv2, decoder_gate2]) 
decoder_2 = Add()([decoder_2i, decoder_glu2])
# 2nd attention
attention_matrix2 = Dot(axes=2)([decoder_2, image_encoder])
attention_softmax2 = Activation('softmax')(attention_matrix2)
decoder_c2 = Dot(axes=2)([attention_softmax2, image_encoderTrans])
decoder_3i = Add()([decoder_2, decoder_c2])
# 3rd block
decoder_3o = concatenate([padding_input, decoder_3i], axis=1)
decoder_conv3 = Conv1D(48, 3, padding='valid')(decoder_3o)
decoder_gate3 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_3o)
decoder_glu3 =  Multiply()([decoder_conv3, decoder_gate3]) 
decoder_3 = Add()([decoder_3i, decoder_glu3])
# 3rd attention
attention_matrix3 = Dot(axes=2)([decoder_3, image_encoder])
attention_softmax3 = Activation('softmax')(attention_matrix3)
decoder_c3 = Dot(axes=2)([attention_softmax3, image_encoderTrans])
decoder_4i = Add()([decoder_3, decoder_c3])
# 4th block
decoder_4o = concatenate([padding_input, decoder_4i], axis=1)
decoder_conv4 = Conv1D(48, 3, padding='valid')(decoder_4o)
decoder_gate4 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_4o)
decoder_glu4 =  Multiply()([decoder_conv4, decoder_gate4]) 
decoder_4 = Add()([decoder_4i, decoder_glu4])
# 4th attention
attention_matrix4 = Dot(axes=2)([decoder_4, image_encoder])
attention_softmax4 = Activation('softmax')(attention_matrix4)
decoder_c4 = Dot(axes=2)([attention_softmax4, image_encoderTrans])
decoder_5i = Add()([decoder_4, decoder_c4])
# 5th block
decoder_5o = concatenate([padding_input, decoder_5i], axis=1)
decoder_conv5 = Conv1D(48, 3, padding='valid')(decoder_5o)
decoder_gate5 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_5o)
decoder_glu5 =  Multiply()([decoder_conv5, decoder_gate5]) 
decoder_5 = Add()([decoder_5i, decoder_glu5])
# 5th attention
attention_matrix5 = Dot(axes=2)([decoder_5, image_encoder])
attention_softmax5 = Activation('softmax')(attention_matrix5)
decoder_c5 = Dot(axes=2)([attention_softmax5, image_encoderTrans])
decoder_6i = Add()([decoder_5, decoder_c5])
# 6th block
decoder_6o = concatenate([padding_input, decoder_6i], axis=1)
decoder_conv6 = Conv1D(48, 3, padding='valid')(decoder_6o)
decoder_gate6 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_6o)
decoder_glu6 =  Multiply()([decoder_conv6, decoder_gate6]) 
decoder_6 = Add()([decoder_6i, decoder_glu6])
# 6th attention
attention_matrix6 = Dot(axes=2)([decoder_6, image_encoder])
attention_softmax6 = Activation('softmax')(attention_matrix6)
decoder_c6 = Dot(axes=2)([attention_softmax6, image_encoderTrans])
decoder_7i = Add()([decoder_6, decoder_c6])
# 7th block
decoder_7o = concatenate([padding_input, decoder_7i], axis=1)
decoder_conv7 = Conv1D(48, 3, padding='valid')(decoder_7o)
decoder_gate7 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_7o)
decoder_glu7 =  Multiply()([decoder_conv7, decoder_gate7]) 
decoder_7 = Add()([decoder_7i, decoder_glu7])
# 7th attention
attention_matrix7 = Dot(axes=2)([decoder_7, image_encoder])
attention_softmax7 = Activation('softmax')(attention_matrix7)
decoder_c7 = Dot(axes=2)([attention_softmax7, image_encoderTrans])
decoder_8i = Add()([decoder_7, decoder_c7])
# 8th block
decoder_8o = concatenate([padding_input, decoder_8i], axis=1)
decoder_conv8 = Conv1D(48, 3, padding='valid')(decoder_8o)
decoder_gate8 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_8o)
decoder_glu8 =  Multiply()([decoder_conv8, decoder_gate8]) 
decoder_8 = Add()([decoder_8i, decoder_glu8])
# 8th attention
attention_matrix8 = Dot(axes=2)([decoder_8, image_encoder])
attention_softmax8 = Activation('softmax')(attention_matrix8)
decoder_c8 = Dot(axes=2)([attention_softmax8, image_encoderTrans])
decoder_9i = Add()([decoder_8, decoder_c8])
# 9th block
decoder_9o = concatenate([padding_input, decoder_9i], axis=1)
decoder_conv9 = Conv1D(48, 3, padding='valid')(decoder_9o)
decoder_gate9 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_9o)
decoder_glu9 =  Multiply()([decoder_conv9, decoder_gate9]) 
decoder_9 = Add()([decoder_9i, decoder_glu9])
# 9th attention
attention_matrix9 = Dot(axes=2)([decoder_9, image_encoder])
attention_softmax9 = Activation('softmax')(attention_matrix9)
decoder_c9 = Dot(axes=2)([attention_softmax9, image_encoderTrans])
decoder_10i = Add()([decoder_9, decoder_c9])
# 10th block
decoder_10o = concatenate([padding_input, decoder_10i], axis=1)
decoder_conv10 = Conv1D(48, 3, padding='valid')(decoder_10o)
decoder_gate10 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_10o)
decoder_glu10 =  Multiply()([decoder_conv10, decoder_gate10]) 
decoder_10 = Add()([decoder_10i, decoder_glu10])
# 10th attention
attention_matrix10 = Dot(axes=2)([decoder_10, image_encoder])
attention_softmax10 = Activation('softmax')(attention_matrix10)
decoder_c10 = Dot(axes=2)([attention_softmax10, image_encoderTrans])
decoder_11i = Add()([decoder_10, decoder_c10])
# 11th block
decoder_11o = concatenate([padding_input, decoder_11i], axis=1)
decoder_conv11 = Conv1D(48, 3, padding='valid')(decoder_11o)
decoder_gate11 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_11o)
decoder_glu11 =  Multiply()([decoder_conv11, decoder_gate11]) 
decoder_11 = Add()([decoder_11i, decoder_glu11])
# 11th attention
attention_matrix11 = Dot(axes=2)([decoder_11, image_encoder])
attention_softmax11 = Activation('softmax')(attention_matrix11)
decoder_c11 = Dot(axes=2)([attention_softmax11, image_encoderTrans])
decoder_12i = Add()([decoder_11, decoder_c11])
# 12th block
decoder_12o = concatenate([padding_input, decoder_12i], axis=1)
decoder_conv12 = Conv1D(48, 3, padding='valid')(decoder_12o)
decoder_gate12 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_12o)
decoder_glu12 =  Multiply()([decoder_conv12, decoder_gate12]) 
decoder_12 = Add()([decoder_12i, decoder_glu12])
# 12th attention
attention_matrix12 = Dot(axes=2)([decoder_12, image_encoder])
attention_softmax12 = Activation('softmax')(attention_matrix12)
decoder_c12 = Dot(axes=2)([attention_softmax12, image_encoderTrans])
decoder_13i = Add()([decoder_12, decoder_c12])
# 13th block
decoder_13o = concatenate([padding_input, decoder_13i], axis=1)
decoder_conv13 = Conv1D(48, 3, padding='valid')(decoder_13o)
decoder_gate13 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_13o)
decoder_glu13 =  Multiply()([decoder_conv13, decoder_gate13]) 
decoder_13 = Add()([decoder_13i, decoder_glu13])
# 13th attention
attention_matrix13 = Dot(axes=2)([decoder_13, image_encoder])
attention_softmax13 = Activation('softmax')(attention_matrix13)
decoder_c13 = Dot(axes=2)([attention_softmax13, image_encoderTrans])
decoder_14i = Add()([decoder_13, decoder_c13])
# 14th block
decoder_14o = concatenate([padding_input, decoder_14i], axis=1)
decoder_conv14 = Conv1D(48, 3, padding='valid')(decoder_14o)
decoder_gate14 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_14o)
decoder_glu14 =  Multiply()([decoder_conv14, decoder_gate14]) 
decoder_14 = Add()([decoder_14i, decoder_glu14])
# 14th attention
attention_matrix14 = Dot(axes=2)([decoder_14, image_encoder])
attention_softmax14 = Activation('softmax')(attention_matrix14)
decoder_c14 = Dot(axes=2)([attention_softmax14, image_encoderTrans])
decoder_15i = Add()([decoder_14, decoder_c14])
# 15th block
decoder_15o = concatenate([padding_input, decoder_15i], axis=1)
decoder_conv15 = Conv1D(48, 3, padding='valid')(decoder_15o)
decoder_gate15 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_15o)
decoder_glu15 =  Multiply()([decoder_conv15, decoder_gate15]) 
decoder_15 = Add()([decoder_15i, decoder_glu15])
# 15th attention
attention_matrix15 = Dot(axes=2)([decoder_15, image_encoder])
attention_softmax15 = Activation('softmax')(attention_matrix15)
decoder_c15 = Dot(axes=2)([attention_softmax15, image_encoderTrans])
decoder_16i = Add()([decoder_15, decoder_c15])
# 16th block
decoder_16o = concatenate([padding_input, decoder_16i], axis=1)
decoder_conv16 = Conv1D(48, 3, padding='valid')(decoder_16o)
decoder_gate16 = Conv1D(48, 3, padding='valid', activation='sigmoid')(decoder_16o)
decoder_glu16 =  Multiply()([decoder_conv16, decoder_gate16]) 
decoder_16 = Add()([decoder_16i, decoder_glu16])
# 16th attention
attention_matrix16 = Dot(axes=2)([decoder_16, image_encoder])
attention_softmax16 = Activation('softmax')(attention_matrix16)
decoder_c16 = Dot(axes=2)([attention_softmax16, image_encoderTrans])
decoder_17i = Add()([decoder_16, decoder_c16])

language_decoder = Dense(128, activation='relu')(decoder_17i)
decoder = concatenate([image_encoderTrans, language_decoder])
decoder = GRU(512, return_sequences=True, name='GRU1')(decoder)
decoder = GRU(512, return_sequences=False, name='GRU2')(decoder)
decoder_softmax = Dense(vocab_size, activation='softmax', name='softmax')(decoder) 

#model
my_model = Model(inputs=[image_input, language_input, padding_input], outputs=decoder_softmax)
my_model.load_weights("/rap_blues/lunwen/convs2s/kai/layer16/220/org-weights-epoch-06--val_loss-0.4809--loss-0.0084.hdf5")


json_file = open('/rap_blues/lunwen/convs2s/kai/architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("/rap_blues/lunwen/convs2s/kai/layer16/0/org-weights-epoch-195--val_loss-0.5784--loss-0.0275.hdf5")
my_model.get_layer('conv1').set_weights(model.get_layer('conv1').get_weights())
my_model.get_layer('conv2').set_weights(model.get_layer('conv2').get_weights())
my_model.get_layer('conv3').set_weights(model.get_layer('conv3').get_weights())
my_model.get_layer('conv4').set_weights(model.get_layer('conv4').get_weights())
my_model.get_layer('conv5').set_weights(model.get_layer('conv5').get_weights())
my_model.get_layer('conv6').set_weights(model.get_layer('conv6').get_weights())
my_model.get_layer('conv7').set_weights(model.get_layer('conv7').get_weights())
my_model.get_layer('conv9').set_weights(model.get_layer('conv9').get_weights())
my_model.get_layer('GRU1').set_weights(model.get_layer('GRU1').get_weights())
my_model.get_layer('GRU2').set_weights(model.get_layer('GRU2').get_weights())




#train
optimizer = Adam(lr=0.0001, clipvalue=1.0)
my_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
filepath="/rap_blues/lunwen/convs2s/kai/layer16/220/org-weights-epoch-{epoch:02d}--val_loss-{val_loss:.4f}--loss-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=5)
reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=0, mode='min', epsilon=0.0001, cooldown=0, min_lr=0)  
callbacks_list = [checkpoint, reduceLR]
my_model.fit([image_data, X, padding_up], y, batch_size=128, shuffle=False, validation_split=0.1, callbacks=callbacks_list, verbose=1, epochs=200)

#save the model architecture
model_json = my_model.to_json()  
with open("/rap_blues/lunwen/convs2s/kai/architecture.json", "w") as json_file:  
    json_file.write(model_json)














