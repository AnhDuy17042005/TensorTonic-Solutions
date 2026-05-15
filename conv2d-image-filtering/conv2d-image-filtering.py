import numpy as np

def compute_output_dim(H, W, p, s, k_size):
    """
    Compute output dimentions
    """
    H_out = (H + 2*p - k_size) // s + 1
    W_out = (W + 2*p - k_size) // s + 1
    return H_out, W_out


def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    image  = np.array(image, dtype=np.float32)
    kernel = np.array(kernel, dtype=np.float32)   
    s = int(stride)
    p = int(padding)

    H, W = image.shape
    k_size = kernel.shape[0]

    H_out, W_out = compute_output_dim(H, W, p, s, k_size)
    output = np.zeros((H_out, W_out), dtype=np.float32)

    padded = np.pad(
        image, 
        pad_width=p, 
        mode="constant", 
        constant_values=0
    )

    for i in range(H_out):
        for j in range(W_out):
            start_i = i * s
            start_j = j * s

            pad = padded[start_i: start_i + k_size, start_j: start_j + k_size]
            output[i][j] = np.sum(pad * kernel)
    
    return output.tolist()