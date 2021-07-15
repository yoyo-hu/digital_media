import tkinter
from tkinter import *

import cv2
import tkinter.filedialog
import face_swap
import face_detect
from PIL import Image, ImageTk
from torchvision import transforms as transforms

# 创建界面窗口
root = Tk()
e = tkinter.StringVar()
isLoadPicAndVideo = 0  # 是否已经加载照片和视频，没有加载照片不进行操作

## 实现在本地电脑选择图片和视频或者生成换脸视频结果
def ChoosePicFileOrGetResult(option):
    # 在本地电脑选择未被换脸的图片文件
    def ChooseFile():
        select_file = tkinter.filedialog.askopenfilename(title='选择图片')
        e.set(select_file)

    # 启动换脸程序并将换脸结果保存
    def swapPicandVideo():
        if srcPoints is None:
            print('No face detected in the source image !!!')
            exit(-1)
        str1 = picPth[-6:-4]
        str2 = videoPth[-6:-4]
        #使用cv2.VideoWriter初始化并生成写入对象，写入的目的路径文件为指定的filename的值。
        writer = cv2.VideoWriter(
            filename="./result_videos/from_p{}_and_v{}.mp4".format(str1, str2),
            fourcc=cv2.VideoWriter_fourcc('m', '2', 'v', '1'),
            frameSize=(int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            fps=25.0,
            isColor=True)
        #当视频还在执行时候
        while video.isOpened():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, frame_in = video.read()#读取当前页面的视频帧（图片）
            if ret == True:
                _, dst_img = video.read()#读取当前页面的视频帧（图片）
                #提取该图片中的人脸
                dstPoints, dstShape, dst_face = face_detect.select_face(
                    dst_img, choose=False)
                if dstPoints is not None:
                    #如果存在人脸则进行该图片的换脸
                    dst_img = face_swap.face_swap(
                        srcFace, dst_face, srcPoints, dstPoints, dstShape, dst_img, 68)
                #把换脸结果写入到视频中
                writer.write(dst_img)
                #展现该结果
                cv2.imshow("Video", dst_img)
            else:
                #资源释放
                video.release()
                writer.release()
                cv2.destroyAllWindows()
                break

    if(option == 1):  # 选择视频
        ChooseFile()#选择视频文件
        global video, videoPth
        videoPth = e.get()#读取视频路径
        video = cv2.VideoCapture(e.get())#读取视频文件
        global isLoadPicAndVideo
        isLoadPicAndVideo += 1#表示已经加载好时评

    if(option == 2): # 选择图片
        ChooseFile()#选择文件
        global srcImg, srcPoints, srcShape, srcFace, picPth
        picPth = e.get()#读取图片路径
        srcImg = cv2.imread(e.get())#读取图片文件
        srcPoints, srcShape, srcFace = face_detect.select_face(srcImg)
        if srcPoints is None:#当图片中没有检测到脸部时，打印no face并退出程序
            print('No face')
            exit(-1)
        isLoadPicAndVideo += 1

    if(option == 3 and isLoadPicAndVideo == 2):  
        # 在已经选择源头文件的基础上，生成换脸视频结果
        swapPicandVideo()


Label(root, text="先选择视频文件，再选择图片文件，才能得出换脸结果").grid(row=1, column=1)
Label(root, text="等待换脸结果出来，再退出程序").grid(row=2, column=1)
Button(root, text="选择视频文件", command=lambda: ChoosePicFileOrGetResult(1)).grid(
    row=3, column=1)
Label(root, text=" ").grid(row=4, column=1)
Button(root, text="选择图片文件", command=lambda: ChoosePicFileOrGetResult(2)).grid(
    row=5, column=1)
Label(root, text=" ").grid(row=6, column=1)
Button(root, text="得到换脸结果", command=lambda: ChoosePicFileOrGetResult(3)).grid(
    row=7, column=1)
Label(root, text=" ").grid(row=8, column=1)
Button(root, text="退出", command=root.quit).grid(row=9, column=1)
Label(root, text=" ").grid(row=10, column=1)
root.mainloop()
