import os, glob, shutil
'''
해당 경로 안에 특정 라벨을 수정할때 사용되는 코드
'''
normal_classes_path = '/Users/kpst/Desktop/labelImg-master/data/classes.txt'  #정상 classes.txt 경로
copy_dir = '/Users/kpst/Desktop/권순우_인턴/220419_문화재/세부도' #확인할 경로

dir_list = glob.glob(copy_dir + '/*/*/*') #classes.txt 수정할 경로
file_name = 'classes.txt'

for dir in dir_list:
    final_name = os.path.join(dir,file_name)
    try:
        # print(dir)
        shutil.copy(normal_classes_path,final_name)
    except shutil.SameFileError:
        pass
    # print(final_name)

print('done')