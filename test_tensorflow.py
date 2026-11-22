# Construction Safety AI Project
# Verify TensorFlow installation and Python environment

import sys
import tensorflow as tf
import numpy as np
import matplotlib

print("Python version:", sys.version)
print("Python executable:", sys.executable)

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Matplotlib version:", matplotlib.__version__)

# Create a simple TensorFlow tensor
test_tensor = tf.constant([[1, 2], [3, 4]])

print("\nTest Tensor:")
print(test_tensor.numpy())

print("\nTensorFlow installation successful.")
