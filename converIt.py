import cv2
import os
import sys

def main():
    if len(sys.argv) != 4:
        print("Invalid number of arguments provided")
        sys.exit(1)
    
    src_dir=sys.argv[1]
    des_dir=sys.argv[2]
    limit=int(sys.argv[3])

    if not os.path.isdir(des_dir):
        os.mkdir(des_dir)

    count=0

    for i in os.listdir(src_dir):
        if(".jpg" in i or ".png" in i or ".jpeg" in i):
            file=cv2.imread(src_dir+"/"+i)
            # Converting rgb to grayscale
            file=cv2.cvtColor(file,cv2.COLOR_BGR2GRAY)
            # Sacing the file
            cv2.imwrite(des_dir+"/"+i,file)
            count+=1
        if(limit !=-1 and count==limit):
            break
   
if __name__=="__main__":
    main()