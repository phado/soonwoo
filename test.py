import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

print("heelo open cv TT",cv2.__version__)

def readtxt2(img,txt):
    # test = '/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/number_399/1/front/front_1.txt'
    src = cv2. imread(img)
    dh, dw, _ = src.shape

    fl = open(txt, 'r')
    data = fl.readlines()
    fl.close()

    for dt in data:
        _, x, y, w, h = map(float, dt.split(' '))

        startx = int((x - w / 2) * dw)
        lastx = int((x + w / 2) * dw)
        starty = int((y - h / 2) * dh)
        lasty = int((y + h / 2) * dh)
        
        cv2.rectangle(src, (startx, starty), (lastx, lasty), (0, 0, 255), 1)
    plt.imshow(src)
    plt.show()

data_list = glob.glob('/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/*/*/*/*')
img_list = glob.glob('/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/*/*/*'+'/*.png')
img_list += glob.glob('/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/*/*/*'+'/*.jpg')
txt_list = glob.glob('/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/*/*/*/'+'/*.txt')

for imglist in img_list:
    img_name,ext = os.path.splitext(imglist)
    for txtlist in txt_list:
        txt_name,ext = os.path.splitext(txtlist)

        if img_name == txt_name:
            readtxt2(imglist,txtlist)


   
# img_path = '/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/number_399/1/front/front_1.jpg'
# txt_path = '/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/number_399/1/front/front_1.txt'
# path = '/Users/kpst/Desktop/src.png'

# def readtxt():
#     test = '/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도/number_399/1/front/front_1.txt'
#     count = 0
#     i=0
#     with open(test) as f:
#         for line in f:
#             lines = line
#             count+=1
#         rows = count
#         cols = 4
#         # result = [[0 for j in range(cols)] for i in range(rows)]
#         # result = np.array([[0 for j in range(cols)] for i in range(rows)])
#         result = np.zeros(shape = (rows,cols))
#     with open(test)as f:    
        
        
#         for i in range(rows):
#             for line in f:
#                 lines = line
                
#                 cntx = lines.split(' ')[1]
#                 fcntx = float(cntx)
            
#                 cnty = lines.split(' ')[2]
#                 fcnty = float(cnty)
                
#                 width = lines.split(' ')[3]
#                 fwidth = float(width)
#                 fwidth = int(fwidth * dw)
                
#                 height = lines.split(' ')[4]
#                 fheight = float(height)
#                 fheight = int(fheight*dh)
                    
#                 startxx = (fcntx-(fwidth/2)*dw)
#                 startx =  int(startxx)
                
#                 startyy = fcnty + ((fheight/2)*dh)
#                 starty = int(startyy)
            
#                 # result = np.append(result, np.array([[startx, starty, fwidth, fheight]]), axis=0)
#                 result[i] = [startx, starty, fwidth, fheight]
#                 i+=1
                
#         return result



    

# imread로 이미지 읽기
# src = cv2. imread(img_path)
# dh, dw, _ = src.shape  
# try:
    # cv2.resize(img,dsize,fx,fy,interpolation)
    # arg1:image, arg2:가로세로 형태의 tuple, arg3,4:가로 세로 사이즈의 배수, arg5:보간법
#     src = cv2.resize(src, None, fx=1.5, fy=1.5)
# except:
#     print ("error")
# dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# (_)가 독립적으로 사용되는 경우 -> 인터프리터에서 마지막 실행결과 값을 가지는 변수로 사용됨
# (_)가 변수로 사용되는 경우 -> 변수 값을 굳이 사용할 필요가 없을 때 사용 ex) c언어의 for문 중 i
# (__이름__) -> 내장된 특수한 함수와 변수를 나타낸다 ex) __init__, __add__

# 이미지 임계처리(이진화)
# threshold(src, thresh, maxcal, type) = 이진화 함수
# arg : (input image, 임계값, maxval(임계값을 넘었을 때 작용함), type(threshhold type))
# _, th = cv2.threshold(dst, 0 ,225, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
# dst = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

# 레이블 맵을 생성해 객체 정보를 함께 반환하는 함수
# 반환값  
# label : 객체에 번호가 지정된 레이블 맵 
# stats : N행 5열, N은 객체 수 +1이며 각각의 행은 번호가 지정된 객체를 의미, 각각 열에는 x, y, width, height, area
#                                                                     (x,y는 상단 좌표)   (area는 면적)
# centroids : N행 2열, 2열에는 x, y 무게중심 좌표가 입력 되어있음(x의 좌표를 다 더해서 갯수로 나눈 값)
# _, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
# print(labels)
# print(stats)
# print(centroids)
# readtxt2()

#직사각형을 그림

# for x, y,w, h in statss:
    # if (h, w)<dst.shape:
        # reactangle(원본이미지,시작점(x,y),종료점(x,y), (B,G,R), 두께,선형,굵기)
    # cv2.rectangle(dst, (x, y, w, h),(0,255,0),1)

# cv2.imshow('src', src)
# cv2.imshow('th', th)
# cv2.imshow('dst', dst)





            
            


