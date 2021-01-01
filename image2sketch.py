# -*- coding: utf-8 -*-


"""
Edge Detection Algorithms: sobel, canny, Laplacian, LoG, DoG, XDoG

Authors: Danran Chen
All rights are reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def fixed_thresh(value, threshold):
    '''
    固定阈值对单个像素点的二值化
    :param value: 像素灰度值
    :param threshold: 固定阈值
    :return value: 二值化后的像素灰度值
    '''
    if value >= threshold:
        value = 255
    else:
        value = 0
    return value


def sobel(img, threshold=100):
    '''
    利用Sobel算子的边缘检测
    :param img: 灰度图数组
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    img = (img - img.min()) * 255.0 / (img.max() - img.min())
    img = np.pad(img, (2, 2), mode='constant', constant_values=0)
    row, col = img.shape
    resultimg = np.zeros([row, col], np.float)

    # 定义Sobel算子
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    for i in range(row - 2):
        for j in range(col - 2):
            grad_x = abs(np.sum(sobel_x * img[i:i + 3, j:j + 3]))
            grad_y = abs(np.sum(sobel_y * img[i:i + 3, j:j + 3]))
            resultimg[i + 1, j + 1] = 255 - (grad_x ** 2 + grad_y ** 2) ** 0.5
            # 二值化
            if threshold > 0:
                resultimg[i + 1, j + 1] = fixed_thresh(resultimg[i + 1, j + 1], threshold)
    resultimg = resultimg[3:row - 3, 3:col - 3]
    return resultimg


def NMS(grad, dx, dy):
    '''
    Canny第四步，非极大值抑制
    '''
    row, col = grad.shape
    nms = np.zeros([row, col])

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if grad[i, j] != 0:
                gradX = dx[i, j]
                gradY = dy[i, j]
                gradTemp = grad[i, j]
                if np.abs(gradY) > np.abs(gradX):
                    w = np.abs(gradX) / np.abs(gradY)
                    grad2 = grad[i - 1, j]
                    grad4 = grad[i + 1, j]
                    if gradX * gradY > 0:
                        grad1 = grad[i - 1, j - 1]
                        grad3 = grad[i + 1, j + 1]
                    else:
                        grad1 = grad[i - 1, j + 1]
                        grad3 = grad[i + 1, j - 1]
                else:
                    w = np.abs(gradY) / np.abs(gradX)
                    grad2 = grad[i, j - 1]
                    grad4 = grad[i, j + 1]
                    if gradX * gradY > 0:
                        grad1 = grad[i + 1, j - 1]
                        grad3 = grad[i - 1, j + 1]
                    else:
                        grad1 = grad[i - 1, j - 1]
                        grad3 = grad[i + 1, j + 1]

                gradTemp1 = w * grad1 + (1 - w) * grad2
                gradTemp2 = w * grad3 + (1 - w) * grad4
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    nms[i, j] = gradTemp
    return nms


def double_thresh(nms, ratiomin, ratiomax):
    '''
    Canny第五步，双阈值选取，弱边缘判断
    '''
    row, col = nms.shape
    dt = np.zeros([row, col])
    threshmin = np.max(nms) * ratiomin
    threshmax = np.max(nms) * ratiomax
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if (nms[i, j] < threshmin):
                dt[i, j] = 0
            elif (nms[i, j] > threshmax):
                dt[i, j] = 1
            elif ((nms[i - 1, j - 1:j + 1] < threshmax).any() or (nms[i + 1, j - 1:j + 1] < threshmax).any()
                  or (nms[i, [j - 1, j + 1]] < threshmax).any()):
                dt[i, j] = 1
    dt = dt[1:row - 1, 1:col - 1]
    return dt


def canny(img, sigma=1.3, kernel_size=(7, 7), ratiomin=0.08, ratiomax=0.5):
    '''
    利用Canny算子的边缘检测
    :param img: 灰度图数组
    :param sigma: 高斯核函数中参数sigma    
    :param kernel_size: 高斯核大小(height,width)
    :param ratiomin: 低阈值比例
    :param ratiomax: 高阈值比例
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    # 1.转灰度图（输入已转）
    img = (img - img.min()) * 255.0 / (img.max() - img.min())

    # 2.高斯滤波
    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape
    filterimg = np.zeros([row, col], np.float)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            filterimg[i + cH, j + cW] = np.sum(img[i:i + H, j:j + W] * Gaussian_kernel(H, W, sigma))
    filterimg = filterimg[H - 1:row - H + 1, W - 1:col - W + 1]

    # 3.计算梯度值和方向（利用Sobel算子）
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filterimg = np.pad(filterimg, (2, 2), mode='constant', constant_values=0)
    row, col = filterimg.shape
    dx = np.zeros([row, col], np.float)
    dy = np.zeros([row, col], np.float)
    grad = np.zeros([row, col], np.float)
    for i in range(row - 2):
        for j in range(col - 2):
            dx[i + 1, j + 1] = abs(np.sum(sobel_x * filterimg[i:i + 3, j:j + 3]))
            dy[i + 1, j + 1] = abs(np.sum(sobel_y * filterimg[i:i + 3, j:j + 3]))
            grad[i + 1, j + 1] = (dx[i + 1, j + 1] ** 2 + dy[i + 1, j + 1] ** 2) ** 0.5
    dx = dx[1:row - 1, 1:col - 1]
    dy = dy[1:row - 1, 1:col - 1]
    grad = grad[1:row - 1, 1:col - 1]

    # 4.非极大值抑制：取像素点梯度方向的局部梯度最大值（在梯度更大的方向利用插值）
    nms = NMS(grad, dx, dy)  # 填充了(1,1)的0

    # 5.双阈值的选取，弱边缘判断
    resultimg = double_thresh(nms, ratiomin, ratiomax)
    resultimg = 255 - resultimg
    resultimg = (resultimg - resultimg.min()) * 255.0 / (resultimg.max() - resultimg.min())
    row, col = resultimg.shape
    resultimg = resultimg[1:row - 1, 1:col - 1]
    return resultimg


def laplacian(img, operator=1, threshold=225):
    '''
    利用拉普拉斯算子的边缘检测
    :param img: 灰度图数组
    :param operator: {1:四邻域拉普拉斯算子, 2:八邻域拉普拉斯算子}
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    img = (img - img.min()) * 255.0 / (img.max() - img.min())
    img = np.pad(img, (2, 2), mode='constant', constant_values=0)
    row, col = img.shape
    resultimg = np.zeros([row, col], np.float)

    # 定义拉普拉斯算子
    if operator == 1:
        laplacian_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif operator == 2:
        laplacian_operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    for i in range(row - 2):
        for j in range(col - 2):
            resultimg[i + 1, j + 1] = 255 - abs(np.sum(laplacian_operator * img[i:i + 3, j:j + 3]))
            # 二值化
            if threshold > 0:
                resultimg[i + 1, j + 1] = fixed_thresh(resultimg[i + 1, j + 1], threshold)
    resultimg = resultimg[3:row - 3, 3:col - 3]
    return resultimg


def LaplaGauss_kernel(H, W, sigma):
    '''
    生成Laplacian of Gaussian核
    :param H: 高斯核高度
    :param W: 高斯核宽度
    :param sigma: 高斯核函数中参数sigma
    :return kernel: 高斯核
    '''
    cH = H // 2
    cW = W // 2
    kernel = np.zeros([H, W], np.float)
    for x in range(-cH, H - cH):
        for y in range(-cW, W - cW):
            norm2 = x ** 2 + y ** 2
            sigma2 = sigma ** 2
            kernel[x + cH, y + cW] = (norm2 / sigma2 - 2) * np.exp(-norm2 / (2 * sigma2))
    kernel /= kernel.sum()
    return kernel


def LoG(img, sigma=1.3, kernel_size=(7, 7), threshold=100):
    '''
    利用Laplacian of Gaussian算子的边缘检测
    :param img: 灰度图数组
    :param sigma: 高斯核函数中参数sigma
    :param kernel_size: 高斯核大小(height,width)
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    img = (img - img.min()) * 255.0 / (img.max() - img.min())

    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape

    resultimg = np.zeros([row, col], np.float)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            resultimg[i + cH, j + cW] = np.sum(img[i:i + H, j:j + W] * LaplaGauss_kernel(H, W, sigma))
            # 二值化
            if threshold > 0:
                resultimg[i + cH, j + cW] = fixed_thresh(resultimg[i + cH, j + cW], threshold)
    resultimg = resultimg[H:row - H, W:col - W]
    return resultimg


def Gaussian_kernel(H, W, sigma):
    '''
    生成高斯核
    :param H: 高斯核高度
    :param W: 高斯核宽度
    :param sigma: 高斯核函数中参数sigma
    :return kernel: 高斯核
    '''
    cH = H // 2
    cW = W // 2
    kernel = np.zeros([H, W], np.float)
    for x in range(-cH, H - cH):
        for y in range(-cW, W - cW):
            kernel[x + cH, y + cW] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    kernel /= kernel.sum()
    return kernel


def DoG(img, k=1.6, sigma=1, kernel_size=(5, 5), threshold=-1):
    '''
    利用Difference-of-Gaussians算子的边缘检测
    :param img: 灰度图数组
    :param k: DoG中参数k
    :param sigma: 高斯核函数中参数sigma
    :param kernel_size: 高斯核大小(height,width)
    :param threshold: 二值化阈值，负数表示不进行阈值化
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    img = (img - img.min()) * 255.0 / (img.max() - img.min())

    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape

    resultimg = np.zeros([row, col], np.float)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            G = Gaussian_kernel(H, W, sigma)
            Gk = Gaussian_kernel(H, W, k * sigma)
            resultimg[i + cH, j + cW] = 255 - np.sum(img[i:i + H, j:j + W] * (G - Gk))
            # 二值化
            if threshold > 0:
                resultimg[i + cH, j + cW] = fixed_thresh(resultimg[i + cH, j + cW], threshold)
    resultimg = resultimg[H:row - H, W:col - W]
    return resultimg


def soft_thresh(value, threshold, phi):
    '''
    对单个像素点的软阈值化
    :param value: 像素灰度值
    :param threshold: 固定阈值
    :param phi: Larger or smaller ϕ control the sharpness of the black/white transitions in the image
    :return value: 二值化后的像素灰度值
    '''
    if value >= threshold:
        value = 1
    else:
        value = 1 + np.tanh(phi * (value - threshold))
    return value


def XDoG(img, p=45, k=1.6, sigma=1, kernel_size=(5, 5), threshold=100, phi=0.025):
    '''
    利用Extended Difference-of-Gaussians算子的边缘检测
    :param img: 灰度图数组
    :param p: 调整两个高斯滤波器的权重
    :param k: 高斯核函数标准差k*sigma
    :param sigma: 高斯核函数中参数sigma
    :param kernel_size: 高斯核大小(height,width)
    :param threshold: 灰度值变为255的阈值
    :param phi: Larger or smaller ϕ control the sharpness of the black/white transitions in the image
    :return resultimg: 经边缘检测，阈值化的线图数组
    '''
    img = (img - img.min()) * 255.0 / (img.max() - img.min())

    H, W = kernel_size
    cH = H // 2
    cW = W // 2
    img = np.pad(img, (H - 1, W - 1), mode='constant', constant_values=0)
    row, col = img.shape

    resultimg = np.zeros([row, col], np.float)
    for i in range(row - H + 1):
        for j in range(col - W + 1):
            G = Gaussian_kernel(H, W, sigma)
            Gk = Gaussian_kernel(H, W, k * sigma)
            resultimg[i + cH, j + cW] = np.sum(img[i:i + H, j:j + W] * ((1 + p) * G - p * Gk))
            resultimg[i + cH, j + cW] = 255 * soft_thresh(resultimg[i + cH, j + cW], threshold, phi)

    resultimg = resultimg[H:row - H, W:col - W]
    return resultimg


if __name__ == "__main__":
    image = Image.open('./test5.jpeg').resize((128, 128))
    img = np.array(image.convert('L'))
    result_img1 = Image.fromarray(np.uint8(sobel(img, threshold=100)))
    result_img2 = Image.fromarray(np.uint8(canny(img, sigma=1.3, kernel_size=(7, 7), ratiomin=0.08, ratiomax=0.5)))
    result_img3 = Image.fromarray(np.uint8(laplacian(img, operator=1, threshold=225)))
    result_img4 = Image.fromarray(np.uint8(LoG(img, sigma=1.3, kernel_size=(7, 7), threshold=100)))
    result_img5 = Image.fromarray(np.uint8(DoG(img, k=1.6, sigma=1, kernel_size=(5, 5), threshold=-1)))
    result_img6 = Image.fromarray(
        np.uint8(XDoG(img, p=45, k=1.6, sigma=1, kernel_size=(5, 5), threshold=100, phi=0.025)))

    fig = plt.figure(figsize=(16, 10))
    p1 = plt.subplot(241)
    p2 = plt.subplot(242)
    p3 = plt.subplot(243)
    p4 = plt.subplot(244)
    p5 = plt.subplot(245)
    p6 = plt.subplot(246)
    p7 = plt.subplot(247)
    p1.set_title('Original')
    p1.imshow(image)
    p1.axis('off')
    p2.set_title('Sobel')
    p2.imshow(result_img1, cmap='gray')
    p2.axis('off')
    p3.set_title('Canny')
    p3.imshow(result_img2, cmap='gray')
    p3.axis('off')
    p4.set_title('Laplacian')
    p4.imshow(result_img3, cmap='gray')
    p4.axis('off')
    p5.set_title('LoG')
    p5.imshow(result_img4, cmap='gray')
    p5.axis('off')
    p6.set_title('DoG')
    p6.imshow(result_img5, cmap='gray')
    p6.axis('off')
    p7.set_title('XDoG')
    p7.imshow(result_img6, cmap='gray')
    p7.axis('off')
    plt.show()
