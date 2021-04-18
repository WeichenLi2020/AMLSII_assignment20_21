import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19


def PixelLoss(criterion='l1'):
    """pixel loss"""
    if criterion == 'l1':
        return tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))


def ContentLoss(criterion='l1', output_layer=54, before_act=True):
    """content loss"""
    if criterion == 'l1':
        loss_func = tf.keras.losses.MeanAbsoluteError()
    elif criterion == 'l2':
        loss_func = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError(
            'Loss type {} is not recognized.'.format(criterion))
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)

    if output_layer == 22:  # Low level feature
        pick_layer = 5
    elif output_layer == 54:  # Hight level feature
        pick_layer = 20
    else:
        raise NotImplementedError(
            'VGG output layer {} is not recognized.'.format(criterion))

    if before_act:
        vgg.layers[pick_layer].activation = None

    fea_extrator = tf.keras.Model(vgg.input, vgg.layers[pick_layer].output)

    @tf.function
    def content_loss(hr, sr):
        # the input scale range is [0, 1] (vgg is [0, 255]).
        # 12.75 is rescale factor for vgg featuremaps.
        preprocess_sr = preprocess_input(sr * 255.) / 12.75
        preprocess_hr = preprocess_input(hr * 255.) / 12.75
        sr_features = fea_extrator(preprocess_sr)
        hr_features = fea_extrator(preprocess_hr)

        return loss_func(hr_features, sr_features)

    return content_loss

#
def EdgeLoss(criterion='l1'):
    '''calculate edge loss'''

    def _gradient(f):
        '''calcualte gradient of f (image must be in 3 channel)'''
        shape = f.shape
        # calcualte gradient in x axis
        middle = f[:, 2:shape[1], :] - f[:, :shape[1] - 2, :]
        new_shape = [v if i!=1 else 1 for i,v in enumerate(shape)]
        left = tf.reshape((f[:, 1, :] - f[:, 0, :]), new_shape)
        right = tf.reshape((f[:, shape[1] - 1, :] - f[:, shape[1] - 2, :]), new_shape)
        u1 = tf.concat([left, middle, right], axis=1)
        # calcualte gradient in y axis
        middle = f[:, :, 2:shape[2]] - f[:, :, :shape[2] - 2]
        new_shape = [v if i != 2 else 1 for i, v in enumerate(shape)]
        top = tf.reshape((f[:, 1, :] - f[:, 0, :]), new_shape)
        bottom = tf.reshape((f[:, shape[1] - 1, :] - f[:, shape[1] - 2, :]), new_shape)
        u2 = tf.concat([top, middle, bottom], axis=2)
        return u1,u2

    def _transform(u1, u2):
        base = tf.math.sqrt(1 + tf.math.square(u1) + tf.math.square(u2))
        u1 = tf.math.divide(u1,base)
        u2 = tf.math.divide(u2,base)
        return u1, u2

    def _divergence(u1,u2):
        u11,u12 = _gradient(u1)
        u21,u22 = _gradient(u2)
        return u11+u12+u21+u22

    def _soft_edge(img):
        u1, u2 = _gradient(img)
        u1, u2 = _transform(u1,u2)
        return _divergence(u1,u2)

    def edge_loss(hr,sr):
        hr_div = _soft_edge(hr)
        sr_div = _soft_edge(sr)
        if criterion == 'l1':
            return tf.keras.losses.MeanAbsoluteError()(hr_div,sr_div)
        elif criterion == 'l2':
            return tf.keras.losses.MeanSquaredError()(hr_div,sr_div)
        else:
            raise NotImplementedError(
                'Loss type {} is not recognized.'.format(criterion))

    return edge_loss
def DiscriminatorLoss(gan_type='ragan'):
    """discriminator loss"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def discriminator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(hr), sigma(hr - tf.reduce_mean(sr))) +
            cross_entropy(tf.zeros_like(sr), sigma(sr - tf.reduce_mean(hr))))

    def discriminator_loss(hr, sr):
        real_loss = cross_entropy(tf.ones_like(hr), sigma(hr))
        fake_loss = cross_entropy(tf.zeros_like(sr), sigma(sr))
        return real_loss + fake_loss

    if gan_type == 'ragan':
        return discriminator_loss_ragan
    elif gan_type == 'gan':
        return discriminator_loss
    else:
        raise NotImplementedError(
            'Discriminator loss type {} is not recognized.'.format(gan_type))


def GeneratorLoss(gan_type='ragan'):
    """generator loss"""
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    sigma = tf.sigmoid

    def generator_loss_ragan(hr, sr):
        return 0.5 * (
            cross_entropy(tf.ones_like(sr), sigma(sr - tf.reduce_mean(hr))) +
            cross_entropy(tf.zeros_like(hr), sigma(hr - tf.reduce_mean(sr))))

    def generator_loss(hr, sr):
        return cross_entropy(tf.ones_like(sr), sigma(sr))

    if gan_type == 'ragan':
        return generator_loss_ragan
    elif gan_type == 'gan':
        return generator_loss
    else:
        raise NotImplementedError(
            'Generator loss type {} is not recognized.'.format(gan_type))
