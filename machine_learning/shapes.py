# import sys
# import numpy as np
# import cv2 as cv
#
# hsv_min = np.array((0, 54, 5), np.uint8)
# hsv_max = np.array((187, 255, 253), np.uint8)
#
# if __name__ == '__main__':
#     fn = 'squares.jpg'  # имя файла, который будем анализировать
#     # fn = 'rectangle.jpg'  # имя файла, который будем анализировать
#     img = cv.imread(fn)
#
#     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # меняем цветовую модель с BGR на HSV
#     thresh = cv.inRange(hsv, hsv_min, hsv_max)  # применяем цветовой фильтр
#     cv.imshow('contours', thresh)
#     cv.waitKey()
#     contours0, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#
#     # перебираем все найденные контуры в цикле
#     for cnt in contours0:
#         rect = cv.minAreaRect(cnt)  # пытаемся вписать прямоугольник
#         box = cv.boxPoints(rect)  # поиск четырех вершин прямоугольника
#         box = np.int0(box)  # округление координат
#         cv.drawContours(img, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник
#
#     cv.imshow('contours', img)  # вывод обработанного кадра в окно
#
#     cv.waitKey()
#     cv.destroyAllWindows()

import cv2
font = cv2.FONT_HERSHEY_COMPLEX

init = cv2.imread("squares.jpg")
img = cv2.imread("squares.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("rectangle.jpg", cv2.IMREAD_GRAYSCALE)
_, threshold = cv2.threshold(img, 220, 230, cv2.THRESH_BINARY)
# сравним различные способы бинаризации изображения
# ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(init, [approx], 0, (0, 255, 0), 2)
    x = approx.ravel()[0]
    y = approx.ravel()[1]


cv2.imshow("shapes", init)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()