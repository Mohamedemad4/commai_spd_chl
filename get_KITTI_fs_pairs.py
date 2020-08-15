import os
import cv2
import tqdm
import pykitti
import numpy as np 

#os.mkdir("~/data_new")

#open("/home/q/data_new/t.txt","w+").write("HA") 
basedir = './'
dir_struct={}
for i in os.listdir(basedir):
    if i.startswith("2011_") and not i.endswith(".zip"):
        if i not in dir_struct.keys():
            dir_struct.update({i:[]})
            for d in os.listdir(basedir+i):
                if os.path.isdir(basedir+i+"/"+d):
                    d_n=d.split("_")[-2]
                    dir_struct[i].append(d_n)

print(dir_struct)

print("opening data/train.mp4")
c=cv2.VideoCapture("data/train.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
width = int(c.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height) # not to be confused by numpy's h,w scheme
print("Video Size ",size)
spds=[]
txt_file=open("data/train.txt",'r').read().split("\n")

vw = cv2.VideoWriter("/home/q/data_new/train.mp4", fourcc, 10, size)
frames=int(c.get(cv2.CAP_PROP_FRAME_COUNT))

for i in tqdm.trange(frames):    
    ret,frame=c.read()
    if i%2!=0: #ignore every odd frame downsample the video and train.txt to 10fps
        continue
    if ret==False:
        break
    vw.write(frame)
    spds.append(txt_file[i])

keys=[i for i in dir_struct.keys()]

k=0
d_idx=0

for _ in tqdm.trange(sum([len(dir_struct[i]) for i in dir_struct.keys()])):
    data = pykitti.raw(basedir,keys[k],dir_struct[keys[k]][d_idx])

    d_idx+=1

    if d_idx==len(dir_struct[keys[k]]):
        k+=1
        d_idx=0
    
    for idx in tqdm.trange(len(data.oxts),leave=False):
     
        im_pil=data.get_cam3(idx)
        img_np = np.array(im_pil) 
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_cv2=cv2.resize(img_cv2, None, fx = width/img_np.shape[1], fy = height/img_np.shape[0], interpolation = cv2.INTER_CUBIC)
     
        spd_in_mph=data.oxts[idx].packet.vf*2.237
   
        spds.append(spd_in_mph)
        vw.write(img_cv2)
    


open("/home/q/data_new/train.txt","w+").write("\n".join(str(i) for i in spds))
vw.release()
c.release()
