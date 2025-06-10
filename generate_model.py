import tensorflow as tf
import numpy as np
import json
import os

# === Configuration ===
MODEL_NAME = 'fashion_mnist'
BASE_DIR = 'C:/NTUT/Final_DNN_Inference/dnn-inference-LOZOBO2023'
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TF_MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}.h5')
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}.npz')
MODEL_ARCH_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}.json')

# === Create model directory if it doesn't exist ===
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load and preprocess data ===
def load_and_preprocess_data():
    print("Loading and preprocessing Fashion-MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

# === Build and train model ===
def build_and_train_model(x_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Training model...")
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)
    
    return model

# === Save model and convert to .npz and .json ===
def save_and_convert_model(model):
    # Save to .h5
    model.save(TF_MODEL_PATH)
    print(f"Saved model to {TF_MODEL_PATH}")
    
    # Extract and save weights to .npz
    params = {}
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for i, w in enumerate(weights):
                param_name = f"{layer.name}_{i}"
                params[param_name] = w
    
    np.savez(MODEL_WEIGHTS_PATH, **params)
    print(f"Saved weights to {MODEL_WEIGHTS_PATH}")
    
    # Extract and save architecture to .json
    arch = []
    for layer in model.layers:
        config = layer.get_config()
        info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "config": config,
            "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
        }
        arch.append(info)
    
    with open(MODEL_ARCH_PATH, "w") as f:
        json.dump(arch, f, indent=2)
    print(f"Saved architecture to {MODEL_ARCH_PATH}")
    
    # Verify saved weights
    print("\nVerifying saved weights...")
    loaded_weights = np.load(MODEL_WEIGHTS_PATH)
    for key in loaded_weights.files:
        print(f"{key}: shape={loaded_weights[key].shape}")

# === Main execution ===
if __name__ == "__main__":
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Build and train model
    model = build_and_train_model(x_train, y_train)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save and convert model
    save_and_convert_model(model)
    
    print("\nModel files generated successfully!")
    print(f"Model files location: {MODEL_DIR}")