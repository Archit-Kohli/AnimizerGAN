import os
import gradio as gr
import tensorflow as tf
import numpy as np

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
    cropped_image = tf.image.random_crop(
      image[0], size=[IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):
    # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

    # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image

def preprocess_image_train(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def preprocess_image_test(image):
    image = tf.image.resize(image, [286, 286],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = normalize(image)
    return image

OUTPUT_CHANNELS=3

def downsample(filters, size, strides=2, apply_batchnorm=True, leaky_relu=None, padding=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    if padding:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    else:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=strides, padding='valid',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    if leaky_relu is True:
        result.add(tf.keras.layers.LeakyReLU())
    elif leaky_relu is False:
        result.add(tf.keras.layers.ReLU())
    return result
  
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def resnet_block():
    result = tf.keras.models.Sequential()
    result.add(downsample(256,3,strides=1,apply_batchnorm=True,leaky_relu=False))
    result.add(downsample(256,3,strides=1,apply_batchnorm=True,leaky_relu=None))
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down_stack = [
        downsample(64,7,strides=1,apply_batchnorm=False,leaky_relu=True),
        downsample(128,3,strides=2,apply_batchnorm=True,leaky_relu=True),
        downsample(256,3,strides=2,apply_batchnorm=True,leaky_relu=True),
    ]
    resnet_stack = [
        resnet_block()
    ]*9
    up_stack = [
        upsample(128,3),
        upsample(64,3),
    ]
    final_layer = downsample(3,7,strides=1,apply_batchnorm=False)
    
    x = inputs
    
    for down in down_stack:
        x = down(x)
    
    for res in resnet_stack:
        x = x + res(x)
        
    for up in up_stack:
        x = up(x)
        
    x = final_layer(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    x = inputs
    x = downsample(64,4,strides=2,apply_batchnorm=False,leaky_relu=True)(x)
    x = downsample(128,4,strides=2,apply_batchnorm=False,leaky_relu=True)(x)
    x = downsample(256,4,strides=2,apply_batchnorm=False,leaky_relu=True)(x)
    
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = downsample(512,4,strides=1,apply_batchnorm=False,leaky_relu=True,padding=False)(x)
    x = tf.keras.layers.ZeroPadding2D()(x)
    x = downsample(1,4,strides=1,apply_batchnorm=False,leaky_relu=True,padding=False)(x)
    
    return tf.keras.Model(inputs=inputs,outputs=x)
    

generator_m2f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f2m_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_m_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator_m2f = Generator()
generator_f2m = Generator()

discriminator_f = Discriminator()
discriminator_m = Discriminator()

checkpoint_dir = "https://huggingface.co/spaces/ArchitKohli/AnimizerGAN/tree/main"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_m2f=generator_m2f,
                           generator_f2m=generator_f2m,
                           discriminator_m=discriminator_m,
                           discriminator_f=discriminator_f,
                           generator_m2f_optimizer=generator_m2f_optimizer,
                           generator_f2m_optimizer=generator_f2m_optimizer,
                           discriminator_m_optimizer=discriminator_m_optimizer,
                           discriminator_f_optimizer=discriminator_f_optimizer)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def animize(img):
    img = np.array(img)
    print(img.shape)
    img = preprocess_image_test(np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2])))
    img = tf.expand_dims(img,0)
    prediction = generator_f2m(img)

    # converting to -1 to 1
    prediction = (prediction - np.min(prediction)) / (np.max(prediction) - np.min(prediction))
    prediction = prediction * 2 - 1
    prediction = (prediction[0]*0.5+0.5).numpy()
    # out = tf.image.resize(image, [128, 128],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return prediction

interface = gr.Interface(fn=animize, inputs="image", outputs="image")
interface.launch(share=True)