import cv2
import dlib
import numpy
PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
# 加载面部检测器
detector = dlib.get_frontal_face_detector()
# 加载训练模型并获取面部特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# 人脸检测


def face_detection(img):
    # 让检测器找到每个人脸的边界框。 第二个参数中的 1
    # 表示我们应该对图像进行 1 次上采样。
    # 这将使一切变得更大，并使我们能够检测到更多的面孔。
    # 进行人脸检测，获得人脸框的位置信息
    rects = detector(img, 1)
    if len(rects) > 1:
        raise RuntimeError("Too many faces")
    if len(rects) == 0:
        raise RuntimeError("no faces")
    return rects

# 人脸特征点检测


def face_points_detection(img, bbox: dlib.rectangle):
    # 获取框 d 中面部特征点
    shape = predictor(img, bbox)
    # 循环遍历 68 个面部标志并将它们转换为(x, y)坐标的2元组
    # 返回(x, y)坐标数组，输入图像的每个特征点对应每行的一个x，y坐标
    return numpy.asarray(list([p.x, p.y] for p in shape.parts()), dtype=numpy.int)


def select_face(im, r=10, choose=True):
    faces = face_detection(im)
    if len(faces) == 0:
        return None, None, None
    idx = numpy.argmax([(face.right() - face.left()) *
                       (face.bottom() - face.top()) for face in faces])
    bbox = faces[idx]
    points = numpy.asarray(face_points_detection(im, bbox))
    # 提取开始和结束的 (x, y) 坐标
    # 边界框
    imagWidth, imagHigh = im.shape[:2]

    # 确保边界框坐标在空间范围内
    # 图像的尺寸
    left, top = numpy.min(points, 0)
    right, bottom = numpy.max(points, 0)
    # 计算边界框的宽度和高度
    x, y = max(0, left - r), max(0, top - r)
    w, h = min(right + r, imagHigh) - x, min(bottom + r, imagWidth) - y
    # 返回边界框坐标
    return points - numpy.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]
