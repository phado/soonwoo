import random
import os ,glob
import shutil
from PIL import Image
import cv2
import matplotlib.pyplot as plt

num_augmented_images = 200 #증강해서 추가할 데이터 개수

file_path = '/Users/kpst/Desktop/문화재/권순우_인턴_추가/세부도'
img_list = glob.glob(file_path+'/*/*/*/*.png')
img_list += glob.glob(file_path+'/*/*/*/*.jpg')
txt_list = glob.glob(file_path+'/*/*/*/*.txt')
total_origin_image_num = len(img_list) # 이미지 개수
augment_cnt = 1 #증강 횟수

for i in range(1, num_augmented_images):
    change_picture_index = random.randrange(1, total_origin_image_num-1)
    file_name = img_list[change_picture_index]
    
    origin_image_path =  file_name
    full_names = file_name.split('/')[1:10]
    full_name = '/'+full_names[0]+'/'+full_names[1]+'/'+full_names[2]+'/'+full_names[3]+'/'+full_names[4]+'/'+full_names[5]+'/'+full_names[6]+'/'+full_names[7]+'/'+full_names[8]+'/'
    img_names = os.path.basename(file_name)
    img_name,ext = os.path.splitext(img_names)
    origin_image_path1,ext = os.path.splitext(origin_image_path)
    image = cv2.imread(origin_image_path)
    random_augment = random.randrange(3,4)

    if(random_augment == 1):
        #이미지 좌우 반전
        revers_image = cv2.flip(image,1)
        cv2.imwrite(full_name +img_name + '_'+str(augment_cnt)+ '_revers' + ext, revers_image)
        # revers_image.save(full_name + 'revers_' + str(augment_cnt) + ext)
        shutil.copy(full_name+img_name+'.txt',file_path+'/'+'txt'+'/'+img_name+'.txt')
        shutil.move(file_path+'/'+'txt'+'/'+img_name+'.txt',full_name+img_name+ '_'+str(augment_cnt)+ '_revers' +'.txt')

        src = cv2. imread(full_name+img_name+ '_'+str(augment_cnt)+ '_revers' +ext)
        srctxt = full_name+img_name+ '_'+str(augment_cnt)+ '_revers' +'.txt'
        dh, dw, _ = src.shape
        
        fl = open(srctxt, 'r')
        data = fl.readlines()
        fl.close()

        f = open(srctxt, 'w')
        for dt in data:
            _, x, y, w, h = map(float, dt.split(' '))

            startx = int((x - w / 2) * dw)
            lastx = int((x + w / 2) * dw)
            starty = int((y - h / 2) * dh)
            lasty = int((y + h / 2) * dh)

    
            revers_startx = int(dw-lastx)
            revers_starty = starty
            revers_lastx = int(dw-startx) 
            revers_lasty = lasty


            revers_x = revers_startx+(revers_lastx-revers_startx)/2
            revers_x= round(revers_x/dw,6)

            b = str(revers_x)
            f.write(dt.replace(str(x), b,1))
        f.close()

    if(random_augment ==2):
        #이미지 밝게
        bright_image = cv2.add(image,100)
        cv2.imwrite(full_name + img_name + '_' + str(augment_cnt)+'_bright' + ext, bright_image)
        # bright_image.save(full_name + 'revers_' + str(augment_cnt) + ext)
        shutil.copy(full_name+img_name+'.txt',file_path+'/'+'txt'+'/'+img_name+'.txt')
        shutil.move(file_path+'/'+'txt'+'/'+img_name+'.txt',full_name+img_name+ '_'+str(augment_cnt)+ '_bright' +'.txt')

    if(random_augment == 3):
        #이미지 회전
        dh, dw, _ = image.shape
        rotation_image_conter = cv2.getRotationMatrix2D((dw/2.0,dh/2.0), #회전 중심
                                                        45,#회전 각도(양수는 반시계방향, 음수는 시계방향)
                                                        1)#이미지 배율
        #어파인 변환 함수(src img, 실수형 2x3어파인 변환 행렬,(w,h),출력영상, 보간법,가장자리 픽셀 확장 방식, _)
        image_rotation = cv2.warpAffine(image,rotation_image_conter,(dw,dh))
        plt.imshow(image_rotation)
        plt.show()
        