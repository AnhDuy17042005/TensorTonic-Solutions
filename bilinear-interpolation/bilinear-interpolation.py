import numpy as np

def bilinear_resize(image, new_h, new_w):
    """
    Resize a 2D grid using bilinear interpolation.
    """
    image = np.array(image, dtype=float)

    h, w = image.shape

    results = []

    for i in range(new_h):
        row = []
        
        if new_h == 1:
            src_y = 0
        else:
            src_y = i * (h - 1)/(new_h - 1)

        y0 = int(np.floor(src_y))
        y1 = min(y0+1, h-1)
        # Fractional part
        dy = src_y - y0

        for j in range(new_w):
            if new_w == 1:
                src_x = 0
            else:
                src_x = j * (w - 1)/(new_w - 1)

            x0 = int(np.floor(src_x))
            x1 = min(x0+1, w-1)
            # Fractional part
            dx = src_x - x0

            out = (image[y0][x0]*(1-dy)*(1-dx) 
                   + image[y1][x0]*dy*(1-dx) 
                   + image[y0][x1]*(1-dy)*dx 
                   + image[y1][x1]*dy*dx)
            
            row.append(out)
        results.append(row)

    return results