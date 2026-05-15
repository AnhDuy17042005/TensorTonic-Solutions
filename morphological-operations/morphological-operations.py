import numpy as np

def morphological_op(image, kernel, operation):
    """
    Apply morphological erosion or dilation to a binary image.
    """
    image = np.array(image, dtype=np.float32)
    kernel = np.array(kernel, dtype=np.float32)

    h, w = image.shape
    k_size = kernel.shape[0]

    if operation not in ["dilate", "erode"]:
        return None
    
    padding = k_size // 2
    padded  = np.pad(
        image,
        pad_width=padding,
        mode="constant",
        constant_values=0
    )

    output = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + k_size, j:j+k_size]
            patch = patch[kernel == 1]

            if operation == "erode":
                output[i][j] = 1 if np.all(patch == 1) else 0

            if operation == "dilate":
                output[i][j] = 0 if np.any(patch == 1) == 0 else 1
    
    return output.tolist()