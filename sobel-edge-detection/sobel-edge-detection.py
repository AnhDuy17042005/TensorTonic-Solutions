import numpy as np

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    image = np.array(image, dtype=np.float32)
    h, w = image.shape

    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    Ky = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    k_size = Kx.shape[0]
    padded = np.pad(
        image, 
        pad_width=1,
        mode="constant",
        constant_values=0
    )

    output = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            pad = padded[i:i+k_size, j:j+k_size]

            gx = np.sum(pad * Kx)
            gy = np.sum(pad * Ky)
            output[i][j] = np.sqrt(gx**2 + gy**2)

    return output.tolist()