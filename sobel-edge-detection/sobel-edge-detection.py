import numpy as np

def sobel_edges(image):
    """
    Apply the Sobel operator to detect edges.
    """
    img = np.array(image, dtype=np.float32)
    h, w = img.shape
    
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

    padded = np.pad(img, pad_width=1, mode="constant", constant_values=0)
    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3]

            Gx = np.sum(patch * Kx)
            Gy = np.sum(patch * Ky)

            output[i][j] = np.sqrt(Gx**2 + Gy**2)
    
    return output.tolist()