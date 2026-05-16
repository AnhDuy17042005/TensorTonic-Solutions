import math

def rotate_image(image, angle_degrees):
    """
    Rotate the image counterclockwise by the given angle using nearest neighbor interpolation.
    """
    angle_degrees = math.radians(angle_degrees)

    H = len(image)
    W = len(image[0])

    cy = (H - 1)/2
    cx = (W - 1)/2

    rotated = [[0 for _ in range(W)] for _ in range(H)]

    for i in range(H):
        dy = i - cy
        for j in range(W):
            dx = j - cx

            src_y = cy + dy*math.cos(angle_degrees) + dx*math.sin(angle_degrees)
            src_x = cx - dy*math.sin(angle_degrees) + dx*math.cos(angle_degrees)

            src_y = round(src_y)
            src_x = round(src_x)

            if 0 <= src_y < H and 0 <= src_x < W:
                rotated[i][j] = image[src_y][src_x]

    return rotated