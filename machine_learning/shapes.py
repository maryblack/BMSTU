import cv2
font = cv2.FONT_HERSHEY_COMPLEX

init = cv2.imread("squares.jpg")
img = cv2.imread("squares.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("rectangle.jpg", cv2.IMREAD_GRAYSCALE)
# сравним различные способы бинаризации изображения
# _, threshold = cv2.threshold(img, 219, 230, cv2.THRESH_BINARY)
# threshold = cv2.inRange(img, 220, 255)
# _, threshold = cv2.threshold(img, 220, 230, cv2.THRESH_BINARY_INV)
_, threshold = cv2.threshold(img, 220, 230, cv2.THRESH_TOZERO)
# _, threshold = cv2.threshold(img, 220, 230, cv2.THRESH_TOZERO_INV)
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sqr_rect(cnt):
    x1 = cnt[0][0][0]
    y1 = cnt[0][0][0]
    x2 = cnt[2][0][0]
    y2 = cnt[2][0][1]
    return abs(x2 - x1)*abs(y2-y1)

def isrect(cnt):
    x1 = cnt[0][0][0]
    y1 = cnt[0][0][0]
    x2 = cnt[1][0][0]
    y2 = cnt[1][0][1]
    x3 = cnt[2][0][0]
    y3 = cnt[2][0][0]
    x4 = cnt[3][0][0]
    y4 = cnt[3][0][1]

    delta_x12 = abs()



for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    # print(len(approx))

    if len(approx) >= 4 and sqr_rect(cnt) < 5000 and sqr_rect(cnt) > 250:
        cv2.drawContours(init, [approx], 0, (0, 255, 0), 2)
    # cv2.drawContours(init, [approx], 0, (0, 255, 0), 2)
    # if len(approx) == 14:
    #     new = approx[1:5]
    #     cv2.drawContours(init, [new], 0, (0, 255, 0), 2)

    # if len(approx) == 4:# рисование прямоугольника по 2 углам
    #     angle1 = (approx[0][0][0], approx[0][0][1])
    #     angle2 = (approx[2][0][0], approx[2][0][1])
    #     cv2.rectangle(init, angle1, angle2, (0, 255, 0), 2)

    x = approx.ravel()[0]
    y = approx.ravel()[1]


cv2.imshow("shapes", init)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()