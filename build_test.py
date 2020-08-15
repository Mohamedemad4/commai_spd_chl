import sys
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from commai_spd_chl_keras_datagen import commai_speed_chl_gen

frames=20400#10796
model=load_model(sys.argv[1],custom_objects={"tf":tf})

pred_gen=commai_speed_chl_gen(frames_per_sample=2,mode="stacked",batch_size=4,txt="../data/train.txt",v=sys.argv[2],frame_idx=np.arange(frames).reshape(frames//2,2))
pred_gen.build_video_folder()

spds = model.predict_generator(pred_gen,verbose=1,max_queue_size=85,workers=10,use_multiprocessing=True)
print(spds.shape)
#import pdb;pdb.set_trace()
np.save(open("spds_test.npz",'wb+'),spds)
#spds=np.load(open("spds_test.npz",'rb'),allow_pickle=True).flatten()
f=open(sys.argv[3],'w+')
#import pdb;pdb.set_trace()
for i in tqdm.tqdm(spds.flatten()):
    f.write(str(i)+"\n")
f.close()
