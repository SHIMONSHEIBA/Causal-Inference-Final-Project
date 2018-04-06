import pandas as pd
import pickle
from keras import backend as K
# conda install -c conda-forge keras
from keras.layers import Flatten, Dense, Input, Conv2D, Dropout
from keras.layers import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
import keras

### CNN MODEL ###

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

### config
input_shape = (1,50)
num_classes = 2
batch_size = 15,000
epochs = 10
x_train = 
y_train =
x_test =
y_test =
history = AccuracyHistory()

###



model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

print("model output shape is:", model.output_shape)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


###################debug feltas gap
##########################################################
# all_data0304 = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
#                                                           "Causal-Inference-Final-Project\\importing_change_my_view\\"
#                                                           "all_data_0304.csv",
#                                        index_col=False)
#
#
#
#
# new_delta = pd.read_csv(filepath_or_buffer="C:\\Users\\ssheiba\\Desktop\\MASTER\\causal inference\\"
#                                                           "Causal-Inference-Final-Project\\importing_change_my_view\\"
#                                                           "all submissions comments with label.csv",
#                                        index_col=False)
#
# pkl_file = open('OP_deltas_comments_ids_deltalog.pickle', 'rb')
# OP_deltas_comments_ids_deltalog = pickle.load(pkl_file)
#
# # TODO: change save/load places to self class variables
# pkl_file = open('OP_deltas_comments_ids.pickle', 'rb')
# OP_deltas_comments_ids = pickle.load(pkl_file)
#
# pkl_file = open('OP_deltas_comments_ids_latest.pickle', 'rb')
# OP_deltas_comments_ids_latest = pickle.load(pkl_file)
#
# delta_comments_latest = list()
# delta_comments_new = list()
# delta_comments_log = list()
#
# for sub, com_list in OP_deltas_comments_ids_latest.items():
#     delta_comments_latest += com_list
#
# delta_comments_latest_set = list(set(delta_comments_latest))
# delta_comments_latest_len = len(delta_comments_latest_set)
# print("num of latest is ", delta_comments_latest_len)
#
# for sub, com_list in OP_deltas_comments_ids.items():
#     delta_comments_new += com_list
#
# delta_comments_new_set = list(set(delta_comments_new))
# delta_comments_new_len = len(delta_comments_new_set)
# print("num of new is ", delta_comments_new_len)
#
# for sub, com_list in OP_deltas_comments_ids_deltalog.items():
#     delta_comments_log += com_list
#
# delta_comments_log_set = list(set(delta_comments_log))
# delta_comments_log_len = len(delta_comments_log_set)
# print("num of log is ", delta_comments_log_len)
#
# removed = list(set(delta_comments_latest_set) - set(delta_comments_new_set))
# print(removed)
# print("hello")

##################################################################



