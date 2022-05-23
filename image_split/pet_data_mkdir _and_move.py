import glob
import os
import shutil

input_dir = '/Users/kpst/Desktop/images'
save_dir = '/Users/kpst/Desktop/pet_dataset//'

input_dir_list = glob.glob(input_dir + '/*')

cat_list = []
save_img_list = []

for img in input_dir_list:
    # 파일명 받기
    file_name = os.path.basename(img)
    full_name, ext = os.path.splitext(file_name)
    idx = '_'+ full_name.split('_')[-1]
    name = full_name.strip(idx) #인자로 전달된 문자를 String 왼쪽, 오른쪽에서 제거
    if name not in cat_list:
        os.makedirs(os.path.join(save_dir+name), exist_ok=True)
        cat_list.append(name)
    d = os.path.join(save_dir + name + '/' +file_name)
    shutil.copy(img,os.path.join(save_dir+name))