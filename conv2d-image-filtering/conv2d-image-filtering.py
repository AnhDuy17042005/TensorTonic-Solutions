import numpy as np

def compute_ouput_dim(H, W, k_h, k_w, p, s):
    """
    Compute ouput dimension 
    """
    H_out = (H + 2*p - k_h) // s + 1
    W_out = (W + 2*p - k_w) // s + 1
    return H_out, W_out

def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    s = int(stride)
    p = int(padding)

    H, W = image.shape[:2]
    k_h, k_w = kernel.shape[:2]

    H_out, W_out = compute_ouput_dim(H, W, k_h, k_w, p, s)

    # padded image
    padded = np.pad(
        image, 
        pad_width=padding, 
        mode="constant", 
        constant_values=0
    )

    output = np.zeros((H_out, W_out), dtype=float)

    for i in range(H_out):
        for j in range(W_out):
            start_i = i*s
            start_j = j*s

            region = padded[start_i:start_i + k_h, start_j:start_j + k_w]
            output[i, j] = np.sum(region * kernel)
            
    return output.tolist()