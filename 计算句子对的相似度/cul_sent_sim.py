#导入tkinter模块
from tkinter import *

def calculate_sim():
    # 获得需要计算相似度的句子对
    sentence1 = input1.get()
    sentence2 = input2.get()

    ''' 为了避免计算余弦相似度时出现除以
    0的情况，我们要先判断字符串是否为空 '''
    # 两个字符串都为空，相似度为1，一个字符串为空，另一个不为空，相似度为0
    if(len(sentence1)==0 and len(sentence2)==0 ):
        s='\''+sentence1+'\''+' 和 '+'\''+sentence2+'\''+' 的相似度为： %.4f。\n\n'%1
        txt.insert(END, s)   # 在文本框中显示运算结果
        return
    elif( len(sentence1)==0 or len(sentence2)==0 ):
        s='\''+sentence1+'\''+' 和 '+'\''+sentence2+'\''+' 的相似度为： %.4f。\n\n'%0
        txt.insert(END, s)   # 在文本框中显示运算结果
        input1.delete(0, END)  # 清空非空文本框的输入
        input2.delete(0, END)  # 清空非空文本框的输入
        return

    
    #导入nltk中的word_tokenize函数
    from nltk import word_tokenize

    sentences = [sentence1, sentence2]
    #nltk.word_tokenize(sentenceence)表示对sentenceence进行分词，空格隔开句子中的词
    #对sentences中的每个变量用sentence来表示，对于sentence的分词结果中的每个变量用word来表示
    #temp_texts为二维数组，每一行依次存储每一个句子的分词结果
    temp_texts = [[word for word in word_tokenize(sentence)] for sentence in sentences]

    #构建语料库corpus，即句子中所有出现过的单词及标点（不重复）
    #tempList列表用来存储句子中所有的单词及标点（重复）
    tempList = []
    for text in temp_texts:
        tempList += text
    #set() 函数创建一个无序不重复元素集
    corpus = set(tempList)

    #zip(corpus, range(len(corpus)))将corpus和range(len(corpus))两个列表中的元素对应起来形成一个个元组
    #dict根据该元组建立一个方便映射的可变容器模型，赋给corpus_dict
    #这么做的目的是将语料库中的单词及标点建立数字映射
    corpus_dict = dict(zip(corpus, range(len(corpus))))

    #定义vectortor_rep函数，得到句子的向量（单词在语料词中的编号，该单词出现的次数）
    def get_vector(text, corpus_dict):
        vector = []
        # 对于语料库中的每一个词
        for key in corpus_dict.keys():
            if key in text:
                vector.append((corpus_dict[key], text.count(key))) # 若句子中存在该词
            else:
                vector.append((corpus_dict[key], 0)) # 若句子中存在该词

        vector = sorted(vector, key= lambda x: x[0])

        return vector

    #vector1存放句子1的向量
    #vector2存放句子2的向量
    vector1 = get_vector(temp_texts[0], corpus_dict)
    vector2 = get_vector(temp_texts[1], corpus_dict)
 
    #计算向量的余弦相似度
    #引进math中的sqrt函数
    from math import sqrt
    #函数get_cosine_similarity返回vector1和vector2的向量夹角的余弦值
    def get_cosine_similarity(vector1, vector2):
        inner_product = 0
        square_length_vector1 = 0
        square_length_vector2 = 0
        for tup1, tup2 in zip(vector1, vector2):
            inner_product += tup1[1]*tup2[1]
            square_length_vector1 += tup1[1]**2
            square_length_vector2 += tup2[1]**2

        return (inner_product/sqrt(square_length_vector1*square_length_vector2))

    #用向量的余弦相似度作为句子的相似度
    usimilarity = get_cosine_similarity(vector1, vector2)
    s='\''+sentence1+'\''+' 和 '+'\''+sentence2+'\''+' 的相似度为： %.4f。\n\n'%usimilarity
    txt.insert(END, s)   # 在文本框中显示运算结果
    input1.delete(0, END)  # 清空输入
    input2.delete(0, END)  # 清空输入

'''窗体'''
root = Tk() # 初始化一个根窗体实例 root
root.geometry('460x600') # 设置窗体的大小
root.title('计算句子对的相似度') #设置窗体的标题文字

'''输入框'''
s1 = StringVar()
s2 = StringVar() #告诉编译器存放的为字符串类型
input1 = Entry(root,textvariable=s1) # 接收字符串输入
input1.place(relx=0.05, rely=0.15, relwidth=0.4, relheight=0.05) # 设置输入框显示的相对位置，以及相对长宽属性
input2 = Entry(root,textvariable=s2) 
input2.place(relx=0.55, rely=0.15, relwidth=0.4, relheight=0.05)

'''标签'''
uLabel = Label(root, text='输入需要计算相似度的句子对') #设置标签
uLabel.place(relx=0.1, rely=0.05, relwidth=0.8, relheight=0.1) # 设置标签显示的相对位置，以及相对长宽属性

'''调用 calculate_sim()'''
uButton1 = Button(root, text='计算相似度', command=calculate_sim) # 鼠标点击触发函数Calculate the similarity of sentence pairs的执行，其中按钮上的文本为'计算相似度'
uButton1.place(relx=0.1, rely=0.25, relwidth=0.3, relheight=0.075) # 设置按钮显示的相对位置，以及相对长宽属性

'''退出程序'''
uButton2 = Button(root, text='退出', command=root.quit) # 鼠标点击触发程序的退出，其中按钮上的文本为'退出'
uButton2.place(relx=0.6, rely=0.25, relwidth=0.3, relheight=0.075) # 设置按钮显示的相对位置，以及相对长宽属性

'''文本框'''
txt = Text(root) # 定义文本框
txt.place(relx=0.05,rely=0.4, relwidth=0.9,relheight=0.55) #  设置按钮显示的相对位置，以及相对长宽属性

root.mainloop() # 启动应用程序的主循环，等待鼠标和键盘事件
