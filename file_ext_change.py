import os
import glob

sebudo_dir = '/Users/kpst/Desktop/권순우_인턴_추가/세부도/number_399/3/side'
files = glob.glob(sebudo_dir+'/*.png') 
for x in files: 
    if not os.path.isdir(x): 
        filename = os.path.splitext(x) 
        try: 
            os.rename(x,filename[0] + '.jpg') 
        except: 
            pass