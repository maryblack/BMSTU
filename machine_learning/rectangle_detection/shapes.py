import cv2
import math
import numpy as np

class Contour:
    def __init__(self, cnt):
        self.cnt = cnt
        self.area = cv2.contourArea(cnt)
        self.approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

    def angles_list(self):
        angles_by_points = []
        approx = self.approx
        len_c = len(approx)
        new_cnt = np.vstack((self.approx, self.approx[:2]))
        for i in range(len_c):
            px_1 = new_cnt[i][0][0]
            py_1 = new_cnt[i][0][1]
            px_2 = new_cnt[i + 1][0][0]
            py_2 = new_cnt[i + 1][0][1]
            px_3 = new_cnt[i + 2][0][0]
            py_3 = new_cnt[i + 2][0][1]
            x1, y1 = point_to_vect(px_1, px_2, py_1, py_2)
            x2, y2 = point_to_vect(px_3, px_2, py_3, py_2)
            # print(f'point1: {x1}, {y1}\npoint2: {x2}, {y2}')
            # print(angle(x1, x2, y1, y2))
            angles_by_points.append(([new_cnt[i + 1][0]],angle(x1, x2, y1, y2)))

        return angles_by_points

    def split_contours(self):
        contours = []
        i = 0
        cont = []
        ang = self.angles_list()
        ang.append(ang[0]) # добавляем первый элемент, чтобы замкнуть контур
        while i < len(ang):
            if ang[i][1] < 180:
                cont.append(ang[i][0])
            else:
                contours.append(cont)
                cont = []
            i += 1
        contours.append(cont)
        return contours

    def make_rectangle(self, cont):
        px_1 = cont[0][0][0]
        py_1 = cont[0][0][1]
        px_2 = cont[1][0][0]
        py_2 = cont[1][0][1]
        px_3 = cont[2][0][0]
        py_3 = cont[2][0][1]
        dx = px_3 - px_2
        dy = py_1 - py_2
        px_4 = px_1 + dx
        py_4 = py_3 + dy
        pf = np.array([[px_4, py_4]])
        three = np.vstack((cont))
        res = np.vstack((three, pf))

        return res


def angle(x1, x2, y1, y2) -> float:
    v1_theta = math.atan2(y1, x1)
    v2_theta = math.atan2(y2, x2)
    r = (v2_theta - v1_theta) * (180.0 / math.pi)
    if r < 0:
        r += 360.0
    return r

def point_to_vect(x1, x2, y1, y2):
    return x2-x1, y2-y1


def main():
    init = cv2.imread("rectangle_detection/squares.jpg")
    img = cv2.imread("rectangle_detection/squares.jpg", cv2.IMREAD_GRAYSCALE)

    # img = cv2.imread("rectangle.jpg", cv2.IMREAD_GRAYSCALE)
    # сравним различные способы бинаризации изображения
    # _, threshold = cv2.threshold(img, 219, 230, cv2.THRESH_BINARY)
    # threshold = cv2.inRange(img, 220, 255)
    # _, threshold = cv2.threshold(img, 220, 230, cv2.THRESH_BINARY_INV)
    _, threshold = cv2.threshold(img, 220, 230, cv2.THRESH_TOZERO)
    # _, threshold = cv2.threshold(img, 220, 230, cv2.THRESH_TOZERO_INV)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        contour = Contour(cnt)
        approx = contour.approx

        if len(approx) >= 4 and contour.area < 50000 and contour.area > 20:
            # print('='*10)
            # print(len(approx))
            # print(contour.angles_list())
            splited = contour.split_contours()
            for i in range(len(splited)):
                if len(splited[i]) >= 4:
                    cont = np.vstack((splited[i]))
                    cv2.drawContours(init, [cont], 0, (0, 0, 255), 2)
                if len(splited[i]) == 3:
                    res = contour.make_rectangle(splited[i])
                    cv2.drawContours(init, [res], 0, (0, 0, 255), 2)
            # print('=' * 10)

        x = approx.ravel()[0]
        y = approx.ravel()[1]

    # cv2.imshow("shapes", init)
    cv2.imwrite('result.jpg', init)
    # cv2.imshow("Threshold", threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()









