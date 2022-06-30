import os, glob,shutil

path = '/Volumes/T7_2TB/드론_인명구조/DB/0627_검수테스트/cut/100cut'
target_path = '/Volumes/T7_2TB/드론_인명구조/DB/0627_검수테스트/cut/1_95'

result = 0
a = glob.glob(path+'/*')
num = 0
for i in a:
    if num ==95:
        break
    path1 = glob.glob(i+'/*')
    for j in path1:
        shutil.copy(j,target_path)
    num+=1
print(result)