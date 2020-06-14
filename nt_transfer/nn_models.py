from jax.experimental import stax
from jax.nn.initializers import glorot_normal, normal, ones, zeros, he_normal


def gen_mlp_lenet(output_units = 10, W_initializers_str = 'glorot_normal()', b_initializers_str = 'normal()'):
    """ This is a modern variant of the lenet with relu activation """
    return stax.serial(
    stax.Dense(300, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)), stax.Relu,
    stax.Dense(100, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)), stax.Relu,
    stax.Dense(output_units, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)))

def gen_mlp_lenet_binary(W_initializers_str = 'glorot_normal()', b_initializers_str = 'normal()'):
    """ This is a modern variant of the lenet with relu activation """
    return stax.serial(
    stax.Dense(300, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)), stax.Relu,
    stax.Dense(100, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)), stax.Relu,
    stax.Dense(2, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)))

def gen_cnn_lenet_caffe(output_units = 10, W_initializers_str = 'glorot_normal()', b_initializers_str = 'normal()'):
    return stax.serial(
      stax.Conv(out_chan = 20, filter_shape = (5, 5), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Conv(out_chan = 50, filter_shape = (5, 5), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Flatten, 
      stax.Dense(500, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)),
      stax.Relu,
      stax.Dense(output_units, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str))) 


def gen_cnn_lenet_caffe_binary( W_initializers_str = 'glorot_normal()', b_initializers_str = 'normal()'):
    return stax.serial(
      stax.Conv(out_chan = 20, filter_shape = (5, 5), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Conv(out_chan = 50, filter_shape = (5, 5), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Flatten, 
      stax.Dense(500, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)),
      stax.Relu,
      stax.Dense(2, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str))) 
    
def gen_cnn_conv4(output_units = 10, W_initializers_str = 'glorot_normal()', b_initializers_str = 'normal()'):
    # This is an up-scaled version of the CNN in keras tutorial: https://keras.io/examples/cifar10_cnn/ 
    return stax.serial(
      stax.Conv(out_chan = 64, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.Conv(out_chan = 64, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Conv(out_chan = 128, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.Conv(out_chan = 128, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu,
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Flatten, 
      stax.Dense(512 , W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)),
      stax.Relu,
      stax.Dense(output_units, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)) ) 

def gen_cnn_conv4_dropout(mode = 'test', W_initializers_str = 'glorot_normal()', b_initializers_str = 'normal()'):
    # This is an up-scaled version of the CNN in keras tutorial: https://keras.io/examples/cifar10_cnn/ 
    return stax.serial(
      stax.Conv(out_chan = 64, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str) ),
      stax.Relu, 
      stax.Conv(out_chan = 64, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)),
      stax.Relu, 
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Dropout(rate = 0.75, mode = mode),
      stax.Conv(out_chan = 128, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)),
      stax.Relu, 
      stax.Conv(out_chan = 128, filter_shape = (3, 3), W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)),
      stax.Relu,
      stax.MaxPool((2, 2), strides = (2, 2)),
      stax.Dropout(rate = 0.75, mode = mode ),
      stax.Flatten, 
      stax.Dense(512, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)),
      stax.Dropout(rate = 0.5, mode = mode),
      stax.Relu,
      stax.Dense(10, W_init= eval(W_initializers_str), b_init= eval(b_initializers_str)))


def gen_model_dict():
    model_dict = {}
    model_dict['mlp_lenet_binary'] = gen_mlp_lenet_binary
    model_dict['cnn_lenet_caffe_binary'] = gen_cnn_lenet_caffe_binary
    model_dict['cnn_conv4'] = gen_cnn_conv4
    model_dict['mlp_lenet'] = gen_mlp_lenet
    model_dict['cnn_lenet_caffe'] = gen_cnn_lenet_caffe
    model_dict['cnn_conv4_dropout'] = gen_cnn_conv4_dropout

    return model_dict
    

model_dict = gen_model_dict()