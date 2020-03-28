import tensorflow as tf
import tensorflow.contrib.slim as slim


def CBR(input, channel=64, scope='CBR',is_training=True):
    with tf.variable_scope(scope):
        x = slim.conv2d(input, channel, [3,3], padding='same', activation_fn=None)
        x = tf.layers.batch_normalization(x,training=is_training, name='BN')
        x = tf.nn.relu(x)
        return x

def DBR(input, channel=64, scope='DBR',is_training=True):
    with tf.variable_scope(scope):
        x = slim.conv2d(input, channel, [3, 3], padding='same', rate=2, activation_fn=None)
        x = tf.layers.batch_normalization(x, training=is_training, name='BN')
        x = tf.nn.relu(x)
        return x

def SBNet(input, scope="SBNet",is_training=True):
    with tf.variable_scope(scope):
        '''写的比较难看了'''
        x = CBR(input, scope='CBR_1',is_training=is_training)
        x = DBR(x, scope='DBR_1',is_training=is_training)
        x = CBR(x, scope='CBR_2',is_training=is_training)
        x = CBR(x, scope='CBR_3',is_training=is_training)
        x = DBR(x, scope='DBR_2',is_training=is_training)
        x = CBR(x, scope='CBR_4',is_training=is_training)
        x = CBR(x, scope='CBR_5',is_training=is_training)
        x = CBR(x, scope='CBR_6',is_training=is_training)
        x = DBR(x, scope='DBR_3',is_training=is_training)
        x = CBR(x, scope='CBR_7',is_training=is_training)
        x = CBR(x, scope='CBR_8',is_training=is_training)
        x = DBR(x, scope='DBR_4',is_training=is_training)
        return x

def FEBNet(x1, x2, c, scope='FEBNet',is_training=True):
    with tf.variable_scope(scope):
        x1 = CBR(x1, scope='CBR_1',is_training=is_training)
        x1 = CBR(x1, scope='CBR_2',is_training=is_training)
        x1 = CBR(x1, scope='CBR_3',is_training=is_training)
        y = tf.layers.conv2d(x1, c, [3,3], padding='same',name='conv')
        x1 = tf.concat([x2,y], axis=3)
        x1 = tf.nn.tanh(x1)
        return x1,y

def ABNet(x1,x2,c,scope='ABNet'):
    with tf.variable_scope(scope):
        x1 = tf.layers.conv2d(x1, c, [1,1], padding='same',name='conv')
        y = x1 * x2
        return y

def RBNet(res_noise, noise):
    return noise - res_noise

def ADNet(input,scope='ADNet',is_training=True):
    with tf.variable_scope(scope):
        out = SBNet(input,is_training=is_training)
        out1, out2 = FEBNet(out,input,c=3,is_training=is_training)
        out = ABNet(out1,out2,c=3)
        clean_image = RBNet(out, input)
        return out, clean_image


