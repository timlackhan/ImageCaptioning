from keras.layers.wrappers import TimeDistributed
from keras.layers import ReLU

#image
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

#language
language_input = Input(shape=(max_length,))
Embedding1 = Embedding(vocab_size, 128, input_length=max_length, mask_zero=False)(language_input)
Embedding2 = TimeDistributed(Dense(256))(Embedding1)
Recurrent_layer = GRU(256, return_sequences=True)(Embedding2)

#Multimodal
multimodal = concatenate([encoded_image, Embedding2, language_model])
#Decoder


# Compile the model
my_model = Model(inputs=[visual_input , language_input], outputs=decoder)
my_model.load_weights("/rap_blues/lunwen/baseline/120/org-weights-epoch-0058--val_loss-1.3392--loss-0.0026.hdf5")


LSTM_init = GRU(128, return_sequences=True)(encoded_image)




