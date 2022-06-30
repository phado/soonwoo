import os, glob, shutil
'''
해당 경로 안에 특정 라벨을 수정할때 사용되는 코드
'''
normal_classes_path = '/Users/kpst/Desktop/classes.txt'  #정상 classes.txt 경로
copy_dir = '/Users/kpst/Desktop/대표도_단면_구분220523/national_treasure/대표도/*/*' #확인할 경로

dir_list = glob.glob(copy_dir ) #classes.txt 수정할 경로
file_name = 'classes.txt'

# for dir in dir_list:
#     final_name = os.path.join(dir,file_name)
#     try:
#         # print(dir)
#         shutil.copy(normal_classes_path,final_name)
#     except shutil.SameFileError:
#         pass
#     # print(final_name)


#class가 없는 디렉토리 찾아서 넣기
for dir in dir_list:
    dir_list1 = glob.glob(dir+'/*' )
    for dir1 in dir_list1:
        if dir1 not in 'classes.txt':
            shutil.copy(normal_classes_path, dir)
print('done')