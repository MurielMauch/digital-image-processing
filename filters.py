import numpy as np
from math import sqrt

filter_h1 = [-1, 0, 1, -2, 0, 2, -1, 0, 1]  # filtro de detecção de bordas verticais
filter_h2 = [-1, -2, -1, 0, 0, 0, 1, 2, 1]  # filtro de detecção de bordas horizontais


def apply_filter_h1(img, height, width):
    new_image = np.zeros(shape=(height - 1, width - 1))
    print("Applying h1 filter")

    for column in range(0, width - 1):
        for line in range(0, height - 1):
            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            result = 0

            for i in range(0, 9):
                result += pixel_list[i] * filter_h1[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result  # 1/1

    return new_image


def apply_filter_h2(img, height, width):
    new_image = np.zeros(shape=(height - 1, width - 1))
    print("Applying h2 filter")

    for column in range(0, width - 2):
        for line in range(0, height - 2):

            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            result = 0

            for i in range(0, 9):
                result += pixel_list[i] * filter_h2[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result  # 1/1

    return new_image


def apply_filters_h1_and_h2(img, height, width):
    new_image = np.zeros(shape=(height - 1, width - 1))
    print("Applying h1 with h2 filters")

    for column in range(0, width - 2):
        for line in range(0, height - 2):

            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            result = 0

            for i in range(0, 9):
                result += sqrt(((pixel_list[i] * filter_h1[i]) ** 2) + ((pixel_list[i] * filter_h2[i]) ** 2))

            filter_result = result

            new_image[column][line] = filter_result  # 1/1

    return new_image


def apply_filtro_h3(img, h, w):
    filtro_h3 = [-1, -1, -1, -1, 8, -1, -1, -1, -1]

    new_image = np.zeros(shape=(h - 1, w - 1))
    print("Applying h3 filter")

    for column in range(0, w - 1):
        for line in range(0, h - 1):
            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            filter_result = 0
            result = 0

            for i in range(0, 9):
                result += pixel_list[i] * filtro_h3[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result

    return new_image


def apply_filter_h4(img, h, w):
    first_filter = [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9]
    new_image = np.zeros(shape=(h - 1, w - 1))
    print("Applying h4 filter")

    for column in range(0, w - 1):
        for line in range(0, h - 1):
            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            filter_result = 0
            result = 0

            for i in range(0, 9):
                result += pixel_list[i] * first_filter[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result

    return new_image


def apply_filter_h5(img, h, w):
    h5_filter = [-1, -1, 2, -1, 2, -1, 2, -1, -1]

    new_image = np.zeros(shape=(h - 1, w - 1))
    print("Applying h5 filter")

    for column in range(0, w - 1):
        for line in range(0, h - 1):
            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            filter_result = 0
            result = 0

            for i in range(0, 9):
                result += pixel_list[i] * h5_filter[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result

    return new_image


def apply_filter_h6(img, h, w):
    h6_filter = [2, -1, -1, -1, 2, -1, -1, -1, 2]

    new_image = np.zeros(shape=(h - 1, w - 1))
    print("Applying h6 filter")

    for column in range(0, w - 1):
        for line in range(0, h - 1):
            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            filter_result = 0
            result = 0

            for i in range(0, 9):
                result += pixel_list[i] * h6_filter[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result

    return new_image


def apply_filter_h7(img, h, w):
    h7_filter = [0, 0, 1, 0, 0, 0, -1, 0, 0]

    new_image = np.zeros(shape=(h - 1, w - 1))
    print("Applying h7 filter")

    for column in range(0, w - 1):
        for line in range(0, h - 1):
            pixel_list = []
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])

            filter_result = 0
            result = 0

            for i in range(0, 9):
                result += pixel_list[i] * h7_filter[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result

    return new_image


def apply_filter_h8(img, h, w):
    h8_filter = [0, 0, -1, 0, 0,
                 0, -1, -2, -1, 0,
                 -1, -2, 16, -2, -1,
                 0, -1, -2, -1, 0,
                 0, 0, -1, 0, 0]

    new_image = np.zeros(shape=(h - 2, w - 2))
    print("Applying h8 filter")

    for column in range(0, w - 2):
        for line in range(0, h - 2):
            pixel_list = []
            pixel_list.append(img[column - 2][line - 2])
            pixel_list.append(img[column - 1][line - 2])
            pixel_list.append(img[column][line - 2])
            pixel_list.append(img[column + 1][line - 2])
            pixel_list.append(img[column + 2][line - 2])

            pixel_list.append(img[column - 2][line - 1])
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column + 2][line - 1])

            pixel_list.append(img[column - 2][line])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column + 2][line])

            pixel_list.append(img[column - 2][line + 1])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])
            pixel_list.append(img[column + 2][line + 1])

            pixel_list.append(img[column - 2][line + 2])
            pixel_list.append(img[column - 1][line + 2])
            pixel_list.append(img[column][line + 2])
            pixel_list.append(img[column + 1][line + 2])
            pixel_list.append(img[column + 2][line + 2])

            filter_result = 0
            result = 0

            for i in range(len(h8_filter)):
                result += pixel_list[i] * h8_filter[i]

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result

    return new_image


def apply_filter_h9(img, h, w):
    h9_filter = [1, 4, 6, 4, 1,
                 4, 16, 24, 16, 4,
                 6, 24, 36, 24, 6,
                 4, 16, 24, 16, 4,
                 1, 4, 6, 4, 1]

    new_image = np.zeros(shape=(h - 2, w - 2))
    print("Applying h9 filter")

    for column in range(0, w - 2):
        for line in range(0, h - 2):
            pixel_list = []
            pixel_list.append(img[column - 2][line - 2])
            pixel_list.append(img[column - 1][line - 2])
            pixel_list.append(img[column][line - 2])
            pixel_list.append(img[column + 1][line - 2])
            pixel_list.append(img[column + 2][line - 2])

            pixel_list.append(img[column - 2][line - 1])
            pixel_list.append(img[column - 1][line - 1])
            pixel_list.append(img[column][line - 1])
            pixel_list.append(img[column + 1][line - 1])
            pixel_list.append(img[column + 2][line - 1])

            pixel_list.append(img[column - 2][line])
            pixel_list.append(img[column - 1][line])
            pixel_list.append(img[column][line])
            pixel_list.append(img[column + 1][line])
            pixel_list.append(img[column + 2][line])

            pixel_list.append(img[column - 2][line + 1])
            pixel_list.append(img[column - 1][line + 1])
            pixel_list.append(img[column][line + 1])
            pixel_list.append(img[column + 1][line + 1])
            pixel_list.append(img[column + 2][line + 1])

            pixel_list.append(img[column - 2][line + 2])
            pixel_list.append(img[column - 1][line + 2])
            pixel_list.append(img[column][line + 2])
            pixel_list.append(img[column + 1][line + 2])
            pixel_list.append(img[column + 2][line + 2])

            filter_result = 0
            result = 0

            for i in range(len(h9_filter)):
                result += (pixel_list[i] * h9_filter[i])/256

            if result <= 0:
                filter_result = 0
            elif result >= 255:
                filter_result = 255
            else:
                filter_result = result

            new_image[column][line] = filter_result

    return new_image


def apply_filter_1a(img):
    for entry in img:
        for pixel in entry:
            pixel[0] = 0.393*pixel[0] + 0.769*pixel[1] + 0.189*pixel[2] if 0.393*pixel[0] + 0.769*pixel[1] + 0.189*pixel[2] < 255 else 255
            pixel[1] = 0.349*pixel[0] + 0.686*pixel[1] + 0.168*pixel[2] if 0.349*pixel[0] + 0.686*pixel[1] + 0.168*pixel[2] < 255 else 255
            pixel[2] = 0.272*pixel[0] + 0.534*pixel[1] + 0.131*pixel[2] if 0.393*pixel[0] + 0.769*pixel[1] + 0.189*pixel[2] < 255 else 255
    return img


def apply_filter_1b(img):
    for entry in img:
        for pixel in entry:
            pixel[0] = 0.2989*pixel[0] + 0.5870*pixel[1] + 0.1140*pixel[2]
            pixel[1] = 0.2989*pixel[0] + 0.5870*pixel[1] + 0.1140*pixel[2]
            pixel[2] = 0.2989*pixel[0] + 0.5870*pixel[1] + 0.1140*pixel[2]
        return img



