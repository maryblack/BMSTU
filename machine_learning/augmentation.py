from skimage.transform import rotate
from matplotlib import pyplot as plt
import cv2
import numpy as np


class Image():
    def __init__(self, image):
        self.image = image
        self.titles = ['ImgInit']
        self.imgs = [image]

    def rotation(self, angle):
        """
        Поворот на угол
        :param angle: угол
        :return: экземпляр класса Image
        """
        new_pic = rotate(self.image, angle)
        self.titles.append('ImgRotate')
        self.imgs.append(new_pic)
        cv2.imshow('ImgRotate', new_pic)
        plt.show()
        return Image(new_pic)

    def translation(self, dst_x: int, dst_y:int):
        """
        Смещение изображения
        :param dst_x: по оси х
        :param dst_y: по оси у
        :return: экземпляр класса Image
        """
        rows, cols = self.image.shape[:2]
        M = np.float32([[1, 0, dst_x], [0, 1, dst_y]])
        new_pic = cv2.warpAffine(self.image, M, (cols, rows))
        cv2.imshow('ImgTranslation', new_pic)
        plt.show()
        return Image(new_pic)

    def compression(self, fx, fy):
        """
        Сжатие и растяжение
        :param fx: по оси х
        :param fy: по оси y
        :return: экземпляр класса Image
        """
        new_pic = cv2.resize(self.image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('ImgResize', new_pic)
        plt.show()
        return Image(new_pic)

    def symmetry(self, code):
        """
        Отражение изображение
        :param code: code == 0 - по оси х, code > 0 - по оси у, code < 0 - по обеим осям
        :return:
        """
        new_pic = cv2.flip(self.image, code)
        cv2.imshow('ImgFlip', new_pic)
        plt.show()
        return Image(new_pic)



def main():
    filename = 'wilfred.png'
    img = cv2.imread(filename)
    im = Image(img)
    im.\
        rotation(45).\
        translation(100, 10).\
        compression(0.5,1.5).\
        symmetry(0).\
        symmetry(-1)
    cv2.imshow('ImageInit', im.image)
    # final = cv2.hconcat([im.image, res.image])
    # cv2.imshow('fin', final)
    # cv2.imwrite("./debug.png", final)
    # fin = cv2.hconcat(im.imgs)
    # cv2.imwrite("fin.png", fin)
    cv2.waitKey()
    plt.show()


if __name__ == '__main__':
    main()
