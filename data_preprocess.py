import glob
import os
from natsort import natsorted
path = '/home/rafiqul/data/train/labels/*.txt'
files = glob.glob(path)
label_files=[]
for file in files:
    label_files.append(file)
label_files=natsorted(label_files)

file1 = open("/home/rafiqul/data/labels.csv","a")
file1.truncate(0)
file1.write("Id,Category\n")
for file in label_files:
    with open(file) as f:
        for line in f:
            #print(line.split())
            id=os.path.basename(file)
            id = os.path.splitext(id)[0]
            id=id.split("_")[1]
            result=id+","+line[0]+"\n"
            file1.write(result)



