# This demo shows how to apply deep_dream method to interprete a CNN.

from tensorflow.keras import backend as K

layer_name = 'dense_4' # the layer that is to be analysis. In my case it is just the output layer
filter_index = 0  # can be any integer from 0 to 511, as there are 512 filters in that layer

# define the gradiant of CNN

K.set_learning_phase(0) 
# build a loss function that maximizes the activation
# of the nth filter of the layer considered
layer_output = layer_dict[layer_name].output
loss = layer_output[:,filter_index]
grads = K.gradients(model.output,[model.input])[0]
# compute the gradient of the input picture wrt this loss
#grads = K.gradients(loss, model.input)[0]
# normalization trick: we normalize the gradient
#grads /= (K.sqrt(K.mean(K.square(grads))))
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())
# this function returns the loss and grads given the input picture
iterate = K.function([model.input], [loss, grads])
predict = K.function([model.input], [model.output])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=True,
    rotation_range=0.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    horizontal_flip=False)

#datagen = ImageDataGenerator()
datagen.fit(x_train)

def gen_flow_SC(image, y, datagen, batch_size = 64):
    gen_image = datagen.flow(image, y, batch_size=batch_size, seed=666, shuffle=False)
    while True:
            imagei = gen_image.next()
            yield imagei[0], imagei[1]
            
step = 0.01            
input_img_data_ini = gen_flow_SC(input_img_array.reshape(1,79,79,1), # images has size 79*79
                                     [y_test[ind]], datagen).next()[0]            
input_img_data = input_img_data_ini.copy().astype('float32')
for j in range(2): # do 2 iterations of gradiant ascent
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step
    
diffs = input_img_data.reshape(79,79)-input_img_data_ini.reshape(79,79)      # this is the deep dream image
