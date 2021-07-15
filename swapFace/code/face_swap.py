#! /usr/bin/env python
import cv2
import numpy
import scipy.spatial as spatial

COLOUR_CORRECT_BLUR_FRAC = 0.75
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))

## 3D Transform 双线性插值：在每个图像通道上进行插值
def bilinear_interpolate(img, coords):
    # 变量img: 最多 3 通道图像
    # 变量coords: 2行数组。 第一行 = xcoords，第二行 = ycoords
    # 返回值 与坐标形状相同的内插像素数组

    intCoords = numpy.int32(coords)
    x0, y0 = intCoords
    dx, dy = coords - intCoords

    # 4相邻像素
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    interPixel = top * dy + btm * (1 - dy)

    return interPixel.T

## 所提供点的 ROI 内的 x,y 网格坐标
def grid_coordinates(points):
    #变量points: 生成网格坐标的点
    # 返回值 (x, y) 坐标数组
    xmin = numpy.min(points[:, 0])
    xmax = numpy.max(points[:, 0]) + 1
    ymin = numpy.min(points[:, 1])
    ymax = numpy.max(points[:, 1]) + 1

    return numpy.asarray([(x, y) for y in range(ymin, ymax)
                          for x in range(xmin, xmax)], numpy.uint32)

## 仅在目标图像的 ROI 内（dstPoints 中的点）扭曲来自 src_image 的每个三角形。
def process_warp(srcImg, resultImg, triAffines, dstPoints, delaunay):

    roiCoords = grid_coordinates(dstPoints)
    # 顶点的索引。 -1 如果像素不在任何三角形中
    roiTriIndices = delaunay.find_simplex(roiCoords)

    for simplexIndex in range(len(delaunay.simplices)):
        coords = roiCoords[roiTriIndices == simplexIndex]
        numCoords = len(coords)
        outCoords = numpy.dot(triAffines[simplexIndex],
                              numpy.vstack((coords.T, numpy.ones(numCoords))))
        x, y = coords.T
        resultImg[y, x] = bilinear_interpolate(srcImg, outCoords)

    return None

## 计算每个的仿射变换矩阵,从 dstPoints 到 srcPoints 的三角形 (x,y) 顶点
def triangular_affine_matrices(vertices, srcPoints, dstPoints):
    #变量vertices: 三角形角的三元组索引数组
    # 变量srcPoints: [x, y] 数组指向源图像的地标
    # 变量dstPoints: [x, y] 数组指向目标图像的地标
    # 返回三角形的 2 x 3 仿射矩阵变换
    ones = [1, 1, 1]
    for triIndices in vertices:
        srcTri = numpy.vstack((srcPoints[triIndices, :].T, ones))
        dstTri = numpy.vstack((dstPoints[triIndices, :].T, ones))
        mat = numpy.dot(srcTri, numpy.linalg.inv(dstTri))[:2, :]
        yield mat

# 使用三角测量的面部变形，使用 Delaunay 三角剖分来实现
def warp_image_3d(srcImg, srcPoints, dstPoints, dstShape, dtype=numpy.uint8):
    # 对两幅图像进行三角剖分：
    rows, cols = dstShape[:2]
    resultImg = numpy.zeros((rows, cols, 3), dtype=dtype)
    # 使用了 getTriangleList() 函数cv2.Subdiv2D 类 OpenCV 实现 Delaunay三角测量
    delaunay = spatial.Delaunay(dstPoints)
    triAffines = numpy.asarray(list(triangular_affine_matrices(
        delaunay.simplices, srcPoints, dstPoints)))

    process_warp(srcImg, resultImg, triAffines, dstPoints, delaunay)

    return resultImg

## 绘制凸包
def drawConvexHull(im, points, color):
    # 使用cv2.convexHull获得位置的凸包位置
    points = cv2.convexHull(points)
    # 绘制好多边形后并填充,点的顺序不同绘制出来的凸包也不同
    cv2.fillConvexPoly(im, points, color=color)

## 生成遮罩
def getFaceMask(size, points, erodeFlag=1):
    # 定义是为一张图像和一个标记矩阵生成一个遮罩，它
    # 画出了两个白色的凸多边形：一个是眼睛周围的区域，一
    # 个是鼻子和嘴部周围的区域。之后它由11个像素向
    # 遮罩的边缘外部羽化扩展，可以帮助隐藏任何不连续的区域。

    radius = 10  # kernel size
    kernel = numpy.ones((radius, radius), numpy.uint8)

    mask = numpy.zeros(size, numpy.uint8)
    drawConvexHull(mask, points, 255)
    if erodeFlag:
        mask = cv2.erode(mask, kernel, iterations=1)

    return mask


# 色彩校正
def correct_colours(im1, im2, landmarks1):
    # 用RGB缩放校色
    blurAmount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blurAmount = int(blurAmount)
    if blurAmount % 2 == 0:
        blurAmount += 1
    im1Blur = cv2.GaussianBlur(im1, (blurAmount, blurAmount), 0)
    im2Blur = cv2.GaussianBlur(im2, (blurAmount, blurAmount), 0)

    # 避免被零除错误
    im2Blur = im2Blur + 128 * (im2Blur <= 1.0)
    # 用 im2乘以im1的高斯模糊值然后除以 im2 的高斯模糊值
    return numpy.clip(im2.astype(numpy.float64) * im1Blur.astype(numpy.float64) / im2Blur.astype(numpy.float64), 0, 255).astype(numpy.uint8)


def face_swap(srcFace, dstFace, srcPoints, dstPoints, dstShape, dstImg, end=48):
    h, w = dstFace.shape[:2]
    #cv2.imshow("srcFace", srcFace)
    #cv2.imshow("srcFace", dstFace)
    # 3d 映射
    warpedSrcFace = warp_image_3d(
        srcFace, srcPoints[:end], dstPoints[:end], (h, w))
    # 生成一个遮罩,表示换脸区间
    mask = getFaceMask((h, w), dstPoints)
    maskSrc = numpy.mean(warpedSrcFace, axis=2) > 0
    mask = numpy.asarray(mask * maskSrc, dtype=numpy.uint8)
    # 图像位与运算，扣出遮罩罩住的区间
    warpedSrcFace = cv2.bitwise_and(warpedSrcFace, warpedSrcFace, mask=mask)
    # 图像位与运算，扣出遮罩罩住的区间
    dstFaceMasked = cv2.bitwise_and(dstFace, dstFace, mask=mask)
    # 矫正颜色
    warpedSrcFace = correct_colours(dstFaceMasked, warpedSrcFace, dstPoints)

    # 泊松混合
    # 使用 cv2.seamlessClone以目标脸部作为背景，将罩住并调整后的源脸同色系的嵌入
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    output = cv2.seamlessClone(
        warpedSrcFace, dstFace, mask, center, cv2.NORMAL_CLONE)

    # 返回结果图片
    x, y, w, h = dstShape
    dstImgCopy = dstImg.copy()
    dstImgCopy[y:y + h, x:x + w] = output

    return dstImgCopy
