import numpy as np
from nn_predict import softmax

def test_softmax():
    x = np.array([2.0, 1.0, 0.1])
    y = softmax(x)

    print("Softmax output:", y)
    print("Sum of outputs:", np.sum(y))  # Should be close to 1.0

    assert np.all(y >= 0) and np.all(y <= 1), "Output not in [0,1]"
    assert np.isclose(np.sum(y), 1.0, atol=1e-8), f"Output does not sum to 1: {np.sum(y)}"

if __name__ == "__main__":
    test_softmax()
    print("Test passed successfully!")