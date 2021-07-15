import tkinter
import cv2
import tkinter.filedialog
import face_swap
import face_detect
from PIL import Image, ImageTk
from torchvision import transforms as transforms

# 创建界面窗口
window = tkinter.Tk()
window.title("Smart face change")
window.geometry("1280x780+100+2")
# 设置全局变量
original = Image.new('RGB', (300, 400))
img1 = tkinter.Label(window)
img2 = tkinter.Label(window)
img3 = tkinter.Label(window)
# 路径
e = tkinter.StringVar()
isLoadTwoPic = 0  # 是否已经加载两张照片，没有加载照片不进行操作

# 实现在本地电脑选择图片或者生成换脸图片结果
def ChoosePicFileOrGetResult(option):
    # 在本地电脑选择图片文件
    def ChoosePicFile():
        select_file = tkinter.filedialog.askopenfilename(title='选择图片')
        e.set(select_file)
        load = Image.open(select_file)
        load = transforms.Resize((300, 400))(load)

        global original
        original = load
    # 在界面的(placeX, placeY)显示该照片
    def showPic(placeX, placeY):
        global original
        render = ImageTk.PhotoImage(original)
        img2 = tkinter.Label(window, image=render)
        img2.image = render
        img2.place(x=placeX, y=placeY)
    # 保存照片
    def savePic(saveIm):
        str1 = pic1Pth[-6:-4]
        str2 = pic2Pth[-6:-4]
        cv2.imwrite(
            './result_imgs/from_p{}_and_p{}.jpg'.format(str1, str2), saveIm)
        load = Image.open(
            './result_imgs/from_p{}_and_p{}.jpg'.format(str1, str2))#写入照片到指定的路径中
        load = transforms.Resize((300, 400))(load)

        global original
        original = load

    if(option == 1):  # 两张选择源图片文件
        ChoosePicFile() # 选择头部照片（目的照片）
        showPic(100, 100)#在指定位置展示照片
        global dstImg, dstPoints, dstShape, dstFace, pic1Pth
        pic1Pth = e.get()# 得到选择照片的路径
        dstImg = cv2.imread(e.get())#读取照片
        dstPoints, dstShape, dstFace = face_detect.select_face(dstImg)#提取照片中的脸部

        ChoosePicFile()# 选择脸部照片（源照片）
        showPic(800, 100)#在指定位置展示照片
        global srcImg, srcPoints, srcShape, srcFace, pic2Pth
        pic2Pth = e.get()# 得到选择照片的路径
        srcImg = cv2.imread(e.get())#读取照片
        srcPoints, srcShape, srcFace = face_detect.select_face(srcImg)#提取照片中的脸部
        if srcPoints is None or dstPoints is None:#如果两张照片中有一张没有脸部则打印no face并退出程序
            print('NO Face !!!')
            exit(-1)
        global isLoadTwoPic
        isLoadTwoPic = 1#表示已经成功加载照片

    if(option == 2 and isLoadTwoPic == 1):  # 在已经选择源头文件的基础上，生成换脸图片结果
        output = face_swap.face_swap(
            srcFace, dstFace, srcPoints, dstPoints, dstShape, dstImg)
        savePic(output)
        showPic(450, 450)


# 设置选择图片的按钮
button1 = tkinter.Button(window, text="Select Picture",
                         command=lambda: ChoosePicFileOrGetResult(1))
button1.place(x=600, y=50, width=100, height=40)
button2 = tkinter.Button(window, text="Get Result",
                         command=lambda: ChoosePicFileOrGetResult(2))
button2.place(x=600, y=100, width=100, height=40)

# 设置标签分别为原图像和修改后的图像
label1 = tkinter.Label(window, text="Head")
label1.place(x=250, y=50)

label2 = tkinter.Label(window, text="Face")
label2.place(x=950, y=50)

label2 = tkinter.Label(window, text="Change Face Result")
label2.place(x=600, y=400)

# 设置退出按钮
button0 = tkinter.Button(window, text="Exit", command=window.quit)
button0.place(x=600, y=150, width=100, height=40)
window.mainloop()
