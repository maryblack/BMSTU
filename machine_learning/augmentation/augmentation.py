import os
import cv2
import numpy as np


class Image():
    def __init__(self, image):
        self.image = image

    def rotation(self, angle):
        """
        Поворот на угол
        :param angle: угол
        :return: экземпляр класса Image
        """
        rows, cols = self.image.shape[:2]

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        new_pic = cv2.warpAffine(self.image, M, (cols, rows))
        cv2.imwrite('augmentation/ImgRotate.png', new_pic)
        return Image(new_pic)

    def translation(self, dst_x: int, dst_y: int):
        """
        Смещение изображения
        :param dst_x: по оси х
        :param dst_y: по оси у
        :return: экземпляр класса Image
        """
        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, dst_x], [0, 1, dst_y]])
        new_pic = cv2.warpAffine(self.image, M, (cols, rows))
        cv2.imwrite('augmentation/ImgTranslation.png', new_pic)
        return Image(new_pic)

    def compression(self, fx, fy):
        """
        Сжатие и растяжение
        :param fx: по оси х
        :param fy: по оси y
        :return: экземпляр класса Image
        """
        new_pic = cv2.resize(self.image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('augmentation/ImgResize.png', new_pic)
        return Image(new_pic)

    def symmetry(self, code):
        """
        Отражение изображение
        :param code: code == 0 - по оси х, code > 0 - по оси у, code < 0 - по обеим осям
        :return:
        """
        new_pic = cv2.flip(self.image, code)
        cv2.imwrite('augmentation/ImgFlip.png', new_pic)
        return Image(new_pic)


def main():
    # filename = 'wilfred.png'
    try:
        os.mkdir('augmentation')
    except FileExistsError:
        pass
    filename = 'pic.png'
    img = cv2.imread(filename)
    print(img)
    im = Image(img)
    # задаем последовательность операций над изображением
    fin = im. \
        rotation(30). \
        translation(100, 10). \
        compression(0.5, 1). \
        symmetry(-1)
    cv2.imwrite('augmentation/ImgFinal.png', fin.image)


if __name__ == '__main__':
    main()
