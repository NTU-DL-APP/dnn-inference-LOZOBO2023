import os
import json
import numpy as np
from nn_predict import nn_inference
from utils import mnist_reader

# 檔案路徑配置
YOUR_MODEL_PATH = 'model/fashion_mnist'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_PATH}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_PATH}.json'
OUTPUT_FILE = 'test_acc.txt'

def run_inference():
    print("Starting model_run.py...")
    
    # 檢查輸出檔案
    if os.path.exists(OUTPUT_FILE):
        print(f"Removing existing {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)

    try:
        # 載入測試數據
        print("Loading test data from data/fashion...")
        x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
        print(f"Test data loaded: x_test shape = {x_test.shape}, y_test shape = {y_test.shape}")
        print(f"x_test dtype: {x_test.dtype}, y_test dtype: {y_test.dtype}")
        print(f"x_test min: {x_test.min()}, max: {x_test.max()}, mean: {x_test.mean():.2f}")

        # 檢查模型檔案
        print(f"Checking model files at {MODEL_WEIGHTS_PATH} and {MODEL_ARCH_PATH}")
        if not os.path.exists(MODEL_WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights file {MODEL_WEIGHTS_PATH} not found")
        if not os.path.exists(MODEL_ARCH_PATH):
            raise FileNotFoundError(f"Architecture file {MODEL_ARCH_PATH} not found")

        # 載入權重和架構
        print("Loading weights...")
        weights = np.load(MODEL_WEIGHTS_PATH)
        print(f"Weights loaded: {list(weights.keys())}")
        print("Loading architecture...")
        with open(MODEL_ARCH_PATH, 'r') as f:
            model_arch = json.load(f)
        print(f"Architecture loaded: {model_arch}")

        # 隨機打亂數據
        print("Shuffling test data...")
        indices = np.arange(x_test.shape[0])
        np.random.seed(42)  # 固定隨機種子以確保可重現
        np.random.shuffle(indices)
        x_test_shuffled = x_test[indices]
        y_test_shuffled = y_test[indices]

        # 正規化數據
        print("Normalizing data...")
        normalized_X = x_test_shuffled / 255.0
        print(f"Normalized data shape: {normalized_X.shape}, min: {normalized_X.min():.2f}, max: {normalized_X.max():.2f}")

        # 執行分批推論
        print("Performing inference...")
        batch_size = 1000
        predictions = []
        for i in range(0, len(normalized_X), batch_size):
            batch_X = normalized_X[i:i + batch_size]  # 形狀: (batch_size, 784)
            batch_out = nn_inference(model_arch, weights, batch_X)  # 預期輸出: (batch_size, 10)
            batch_pred = np.argmax(batch_out, axis=-1)  # 直接對 (batch_size, 10) 取 argmax
            predictions.append(batch_pred)
            print(f"Processed batch {i // batch_size + 1}/{len(normalized_X) // batch_size + 1}")
        predictions = np.concatenate(predictions)
        print(f"Predictions shape: {predictions.shape}")

        # 計算準確率
        correct = np.sum(predictions == y_test_shuffled)
        acc = correct / len(y_test_shuffled)
        print(f"Accuracy: {acc:.4f} ({correct}/{len(y_test_shuffled)})")

        # 儲存結果
        print(f"Saving accuracy to {OUTPUT_FILE}")
        with open(OUTPUT_FILE, 'w') as file:
            file.write(str(acc))

        return acc

    except Exception as e:
        print(f"Error in run_inference: {str(e)}")
        raise

if __name__ == "__main__":
    print("Running model_run.py...")
    acc = run_inference()
    print(f"Final accuracy: {acc:.4f}")