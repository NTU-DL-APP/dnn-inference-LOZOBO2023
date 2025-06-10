import numpy as np
import json

# === Activation functions ===
def relu(x):
    """Implement the Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)

def softmax(x):
    """Implement the Softmax activation function with numerical stability."""
    x = np.asarray(x, dtype=np.float64)  # Ensure float64 for precision
    if x.ndim == 1:
        x = x.reshape(1, -1)  # Convert 1D to (1, n)
    
    # Subtract max for numerical stability
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_x)
    
    # Avoid division by zero
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    sum_e_x = np.where(sum_e_x == 0, 1e-10, sum_e_x)  # Prevent zero sum
    
    out = e_x / sum_e_x
    
    if x.ndim == 1:
        out = out.flatten()  # Convert back to 1D if input was 1D
    return out

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