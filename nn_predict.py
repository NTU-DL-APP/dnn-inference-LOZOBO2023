import numpy as np
import json

# === Activation functions ===
def relu(x):
    """Implement the Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)

def softmax(x):
    """Numerically stable softmax with epsilon to avoid division by zero."""
    x = np.asarray(x, dtype=np.float64)
    eps = 1e-8  # small constant to prevent division by zero

    if x.ndim == 1:
        x = x - np.max(x)
        e_x = np.exp(x)
        return e_x / (np.sum(e_x) + eps)
    else:
        x = x - np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x)
        return e_x / (np.sum(e_x, axis=1, keepdims=True) + eps)




# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# Infer TensorFlow h5 model using numpy
# Support only Dense, Flatten, relu, softmax now
def nn_forward_h5(model_arch, weights, data):
    x = data.astype(np.float64)  # Ensure float64 for precision
    for layer in model_arch:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]].astype(np.float64)  # Ensure weight precision
            b = weights[wnames[1]].astype(np.float64)  # Ensure bias precision
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

# You are free to replace nn_forward_h5() with your own implementation 
def nn_inference(model_arch, weights, data):
    return nn_forward_h5(model_arch, weights, data)