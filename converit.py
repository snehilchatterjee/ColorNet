import cv2
import os

src_dir=input("Enter the src dir: ")
des_dir=input("Enter the des dir: ")
limit=int(input("Enter the limit (-1 for all files): "))

if not os.path.isdir(des_dir):
    os.mkdir(des_dir)

count=0

for i in os.listdir(src_dir):
    if(".jpg" in i or ".png" in i or ".jpeg" in i):
        file=cv2.imread(src_dir+"/"+i)
        file=cv2.RGB2GRAY(file)
        cv2.save(des_dir+"/"+i,file)
        count+=1
    if(count==limit):
        break
   