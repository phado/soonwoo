from email.mime import image
import os, glob, shutil
import math
import os
from tokenize import String
import splitfolders
import random,glob,shutil


def renamimg(data_dir,tvt):
	new_images_dir = '/Users/kpst/Desktop/문화재2/images'
	new_val_dir = '/Users/kpst/Desktop/문화재2/labels'

	fullname,ext = os.path.splitext(data_dir)
	if not fullname=='classes':
		t_num = fullname.split('/')[-4]
		gp_num = fullname.split('/')[-3]
		fs_name = fullname.split('/')[-2]
		fname = fullname.split('/')[-1]
		img_final_name = new_images_dir+'/'+tvt+'/'+ t_num +'_'+ gp_num+'_'+fname+ext
		old_txt_dir = '/Users/kpst/Desktop/문화재/권순우_인턴/220419_문화재/세부도/'+t_num+'/'+gp_num+'/'+fs_name+'/'+fname+'.txt'
		os.rename(data_dir,img_final_name) 

		txt_final_name = new_val_dir+'/'+tvt+'/'+ t_num +'_'+ gp_num+'_'+fname+'.txt'
		os.rename(old_txt_dir,txt_final_name)
		print(dir)


def img_move():
	data_dir = glob.glob('/Users/kpst/Desktop/문화재/권순우_인턴/220419_문화재/세부도/*/*/*')

	train = 0.8
	val = 0.1
	test = 0.1

	for path in data_dir:
		print(path)
		img_list = glob.glob(path + '/*.jpg')
		img_list += glob.glob(path + '/*.png')
		random.shuffle(img_list)
		txt_list = glob.glob(path + '/*.txt')

		file_list_cnt = len(img_list)
		train_cnt = math.trunc((file_list_cnt * train) + 0.5)
		val_cnt = math.floor(file_list_cnt*val + 0.5)
		test_cnt = math.floor(file_list_cnt*test + 0.5)

        #예외처리
		if file_list_cnt > train_cnt+val_cnt+test_cnt:
			train_cnt+=1
		elif file_list_cnt < train_cnt+val_cnt+test_cnt:
			train_cnt -=1

            
		if file_list_cnt == train_cnt+val_cnt+test_cnt:
			for file in img_list:
				if train_cnt > 0 :
					renamimg(file,"train")
					train_cnt -=1
					continue
				if val_cnt > 0 :
					renamimg(file,"val")
					val_cnt -=1
					continue
				if test_cnt > 0 :
					renamimg(file,"test")
					test_cnt -=1
					continue		

img_move()
print("a")