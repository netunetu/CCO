# CCO
simulate diffusion equation with TensorFlow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display


def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))

sess = tf.InteractiveSession()

#define equation you need
def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.0, 0.1, 0.0],
                           [0.1, -4., 0.1],
                           [0.0, 0.1, 0.0]])
  return simple_conv(x, laplace_k)
 
N = 250
M = 1000

#Initialization　all one
u_init = np.full([N, M], 1, dtype="float32")
DisplayArray(u_init,  rng=[-0.1, 0.1])

eps = tf.placeholder(tf.float32, shape=())
d = tf.placeholder(tf.float32, shape=())

Ut = tf.Variable(np.full([N, 1], 1, dtype="float32"))
Ut_ = Ut + eps * d * laplace(Ut)
    
step = tf.group(Ut.assign(Ut_))

tf.global_variables_initializer().run()

val = []

#simulation
for i in range(1000):
    step.run({eps:0.0013, d:1})
    s = sess.run(Ut)
    v = s[0, 0]
    u_init[1:N, i] = v
    if i % 50 == 0:
        clear_output()
        DisplayArray(u_init, rng=[-0.1, 0.1])
