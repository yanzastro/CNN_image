# This is a demo code to use CNN to measure a certain parameter.
# The architecture is from arxiv:2005.11819

from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.optimizers import SGD,RMSprop
from keras.constraints import max_norm
from keras.layers import Input, Concatenate, Add
from keras.models import Model

hist2d_image = load_image(imagefile) # this line reads the whole dataset. it needs to be modified in real application.

def splitdata(x, y, test_ratio):
    '''
    This function splits data into training and testing set with test_ratio
    '''
    size = x.shape[0]
    test_ind_interval = int(1./(test_ratio))
    test_num = int(size * test_ratio)
    test_ind = np.arange(test_num)*test_ind_interval
    all_ind = np.arange(size)
    all_ind_shuffled = np.arange(size)
    np.random.shuffle(all_ind_shuffled)
    x = x[all_ind_shuffled]
    y = y[all_ind_shuffled]

    train_ind = np.setdiff1d(all_ind, test_ind)
    x_test = x[test_ind]
    y_test = y[test_ind]
    x_train = x[train_ind]
    y_train = y[train_ind]

    return (x_train, y_train), (x_test, y_test), \
        all_ind_shuffled[train_ind], all_ind_shuffled[test_ind]
        
(x_train, y_train), (x_test, y_test), train_ind, test_ind = splitdata(hist2d_image, M200, 0.2)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
#y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Defining the architecture of the neural network

image_input = Input(shape=(nbins-1, nbins-1, 1))
cov = Conv2D(16, (3, 3), activation='relu')(image_input)
cov = AveragePooling2D(pool_size=(2, 2))(cov)
cov = Conv2D(32, (3, 3), activation='relu')(cov)
cov = AveragePooling2D(pool_size=(2, 2))(cov)
cov = Conv2D(64, (3, 3), activation='relu')(cov)
cov = AveragePooling2D(pool_size=(2, 2))(cov)

out = Flatten()(cov)
out = Dense(200, activation = 'relu')(out)
out = Dropout(0.1)(out)
out = Dense(100, activation = 'relu')(out)
out = Dropout(0.1)(out)
out = Dense(100, activation = 'relu')(out)
out = Dropout(0.1)(out)
out = Dense(20, activation = 'relu')(out)
out = Dense(1, activation = 'relu')(out)
#out = Dense(1, activation = 'linear')(out)

#opt = optimizers.RMSprop(lr=.0005, decay=1e-5)
opt = optimizers.RMSprop(lr=lr, decay=decay)
model = Model(image_input, out)


model.compile(loss='mean_squared_logarithmic_error',
              optimizer=opt,
              metrics=['accuracy'])

# Data generator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
#    rotation_range=20,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=True)


def gen_flow(image_train, image, y, batch_size = 64):
    datagen.fit(image_train)
    gen_image = datagen.flow(image, y, batch_size=batch_size, seed=666, shuffle=False)
    while True:
            imagei = gen_image.next()
            yield imagei[0], imagei[1]


datagen.fit(x_train)
train_flow = gen_flow(x_train, x_train, y_train)
test_flow = gen_flow(x_train, x_test, y_test)


history = model.fit_generator(train_flow, validation_data=test_flow, 
                              steps_per_epoch=50*2, validation_steps=13*2,
                              epochs=4000)
