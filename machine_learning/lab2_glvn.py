import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Picture:
    def __init__(self, image):
        self.img = image

    def split_areas(self, num_x, num_y):
        imgwidth, imgheight = self.img.size
        step_x = round(imgwidth/num_x)
        step_y = round(imgheight/num_y)
        i = 0
        while i <= imgwidth:
            j = 0
            while j <= imgheight:
                box = (i, j, i + step_x, j + step_y)
                a = self.img.crop(box)
                a.save(f'{i}{j}.png')
                print(a)
                j += step_y
            i += step_x

                # try:
                #     o = self.img.crop(area)
                #     o.save(os.path.join(path, "PNG", "%s" % page, "IMG-%s.png" % k))
                # except:
                #     pass
                # k += 1




def main():
    filename = 'pic.png'
    # img = cv2.imread(filename)
    img = Image.open(filename)
    pict = Picture(img)
    pict.split_areas(2,2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()