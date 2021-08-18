classes = {
    "front": 0,
    "back": 1,
    "left": 2,
    "right": 3,
    "f_left": 4,
    "f_right": 5,
    "b_left": 6,
    "b_right": 7,
    "interior": 8,
    "trunk": 9,
    "panel": 10,
    "dirty": 11,
    "labeled": 12,
    "damaged": 13
}

label_to_classes = {val:item for item, val in classes.items()}

def crop(img, bbox):
    X = round(bbox[0])
    Y = round(bbox[1])
    W = round(bbox[2] - X)
    H = round(bbox[3] - Y)
    img = img[Y:Y+H, X:X+W, :]
    return img
