import cv2
import numpy as np




#  OpenCV port of 'TENG' algorithm (Krotkov86)
# double tenengrad(const cv::Mat& src, int ksize)
# {
#     cv::Mat Gx, Gy;
#     cv::Sobel(src, Gx, CV_64F, 1, 0, ksize);
#     cv::Sobel(src, Gy, CV_64F, 0, 1, ksize);
#
#     cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);
#
#     double focusMeasure = cv::mean(FM).val[0];
#     return focusMeasure;
# }
#
# // OpenCV port of 'GLVN' algorithm (Santos97)
# double normalizedGraylevelVariance(const cv::Mat& src)
# {
#     cv::Scalar mu, sigma;
#     cv::meanStdDev(src, mu, sigma);
#
#     double focusMeasure = (sigma.val[0]*sigma.val[0]) / mu.val[0];
#     return focusMeasure;

def glvn(img): # GLVN - отношение глобальной дисперсии к глобальному среднему яркости
    mu, sigma = cv2.meanStdDev(img)
    # print(f'mu:{mu}, sigma: {sigma}')
    focus_measure = sigma[0][0]**2/mu[0][0]

    return focus_measure



def main():
    filename = 'pic.png'
    img = cv2.imread(filename)
    # print(img)
    # src = np.array(img)
    # grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('f4')
    print(glvn(img))
    # print(glvn(grayImg))


if __name__ == '__main__':
    main()