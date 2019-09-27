import cv2
import numpy as np
from PIL import Image
import os
from scipy.signal import convolve2d as conv2


class Picture:
    def __init__(self, image):
        self.img = image

    def split_areas(self, num_x, num_y):
        splited = []
        imgwidth, imgheight = self.img.size
        step_x = round(imgwidth / num_x)
        step_y = round(imgheight / num_y)
        i = 0
        while i < imgwidth:
            j = 0
            while j < imgheight:
                box = (i, j, i + step_x, j + step_y)
                a = self.img.crop(box)
                splited.append((((i, j), (i + step_x, j + step_y)), np.asarray(a.convert('L'))))
                j += step_y
            i += step_x

        return splited


def lv(img, m, n, wx, wy):  # local variance at point (m,n)
    res = 0
    I_mean = 0
    for i in range(0, wx):
        for j in range(0, wy):
            I_mean += img[m + i][n + j]

    I_mean = I_mean / (wx * wy)

    for i in range(0, wx):
        for j in range(0, wy):
            res += (img[m + i][n + j] - I_mean) ** 2

    return res / (wx * wy)


def glvn(img):  # чем меньше значение, тем более размыто изображение
    N, M = img.shape
    wx = 3  # window size x
    wy = 3  # window size x
    lv_mean = 0
    res = 0

    for i in range(N - wx):
        for j in range(M - wy):
            lv_mean += lv(img, i, j, wx, wy)

    lv_mean = lv_mean / (N * M)

    for i in range(N - wx):
        for j in range(M - wy):
            res += (lv(img, i, j, wx, wy) - lv_mean) ** 2

    return res / (N * M)


def sobel(img):
    S_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]
           ]
    S_y = [[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]
           ]
    G_x = conv2(img, S_x)
    G_y = conv2(img, S_y)

    S = np.sqrt(np.square(G_x) + np.square(G_y))

    return S


def teng(img):
    S = sobel(img)
    threshold = np.max(S) * 0.8  # надо подбирать
    res = np.where(S > threshold)

    return np.sum(np.square(res))


def sob_variance(img):
    N, M = img.shape
    S = sobel(img)
    threshold = np.max(S) * 0.8
    res = np.where(S > threshold)
    S_mean = np.sum(res) / (N * M)
    sob_var = np.sum(np.square(np.subtract(res, S_mean)))
    return sob_var


def draw_blur(filename, nx, ny, mode=None):
    init = cv2.imread(filename)
    img = Image.open(filename)
    pict = Picture(img)
    res = pict.split_areas(nx, ny)
    blured_areas = []
    if mode == 'glvn':
        GLVN_val = []
        for el in res:
            GLVN = glvn(el[1])
            blured_areas.append((el[0], GLVN))
            GLVN_val.append(GLVN)

        threshold = max(GLVN_val) * 0.2
        for el in blured_areas:
            if el[1] < threshold:
                cv2.rectangle(init, el[0][0], el[0][1], (0, 0, 255), 2)

            # cv2.imshow("blur", init)
        cv2.imwrite('blured_glvn.jpg', init)

    elif mode == 'tenengrad':
        TENG_val = []
        for el in res:
            TENG = teng(el[1])
            blured_areas.append((el[0], TENG))
            TENG_val.append(TENG)

        threshold = max(TENG_val) * 0.4
        for el in blured_areas:
            if el[1] > threshold:
                cv2.rectangle(init, el[0][0], el[0][1], (0, 0, 255), 2)

        cv2.imwrite('blured_teng.jpg', init)

    elif mode == 'sobel_variance':
        SOB_val = []
        for el in res:
            SOB = teng(el[1])
            blured_areas.append((el[0], SOB))
            SOB_val.append(SOB)

        threshold = max(SOB_val) * 0.2
        for el in blured_areas:
            if el[1] > threshold:
                cv2.rectangle(init, el[0][0], el[0][1], (0, 0, 255), 2)

        cv2.imwrite('blured_sobel.jpg', init)


def main():
    path = os.getcwd()
    filename = 'blur_areas_detection/blur.jpg'
    # filename = 'test.jpg'
    mode = ['glvn', 'tenengrad', 'sobel_variance']
    nx = 10
    ny = 10
    draw_blur(filename, nx, ny, mode[0])
    draw_blur(filename, nx, ny, mode[1])
    draw_blur(filename, nx, ny, mode[2])


if __name__ == '__main__':
    main()
