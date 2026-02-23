import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:

    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):

    d_k = K.shape[1]

    scores = np.dot(Q, K.T)

    scaled_scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scaled_scores)

    output = np.dot(attention_weights, V)

    return output, attention_weights
