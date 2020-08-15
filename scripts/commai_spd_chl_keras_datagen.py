import io
import os
import cv2
import tensorflow as tf
import tqdm
import numpy as np


class commai_speed_chl_gen(tf.keras.utils.Sequence):
    '''
        acsess it this way: #or like percentage interface or whatever
            

            com_gen=commai_speed_chl_gen(n_frames_valid=1000,n_frames_test=1400,n_frames_train=18000,batch_size=32)
            train_idx,test_idx,valid_idx=com_gen.get_indexes()

            test_gen=commai_speed_chl_gen(frame_idx=test_idx)
            train_gen=commai_speed_chl_gen(frame_idx=train_idx)
            valid_gen=commai_speed_chl_gen(frame_idx=valid_idx)

            train_gen.build_video_folder() # run this once to build the video folder

            modes are : temporal,stacked
                temporal returns data in shape (batch_size,frames_per_sample,h,w,1)
                stacked returns in (batch_size,h*frames_per_sample,w,1) 
                optical_flow_dense (batch_size,h,w,3) and y in (batch_size,mean) check self.opticalFlowDense and self.__getitem__ for more info
    '''
    def __init__(self, v="data/train.mp4",txt="data/train.txt",frames_per_sample=3,batch_size=32,
                  n_frames_valid=1000,n_frames_test=1400,n_frames_train=18000,per_train=.7,per_test=.15,per_valid=.15,frame_idx=None,
                  distance_factor=60,db_name="imgs",mode="temporal"):
        self.video = v
        self.txt= txt
        self.frames_per_sample = frames_per_sample
        self.n_frames_valid = n_frames_valid
        self.n_frames_test = n_frames_test
        self.n_frames_train = n_frames_train
        self.mode=mode 
        self.per_train = per_train
        self.per_test = per_test
        self.per_valid = per_valid

        self.batch_size = batch_size
        self.frame_idx=frame_idx
        self.db_name=db_name
        self.vcap=cv2.VideoCapture(self.video)
        self.distance_factor = distance_factor  # see self.get_indexes()

        if not per_test+per_train+per_valid==1:
                raise ValueError("All percentages must add up to 1.")
        
        if not (self.mode=="stacked" or self.mode=="temporal" or self.mode=="optical_flow_dense"):
            raise ValueError("Mode not undrestood:{0}".format(mode))

        if self.mode=="optical_flow_dense" and self.frames_per_sample!=2:
            raise ValueError("optical_flow_dense expects frames_per_sample to be 2 not {0}".format(self.frames_per_sample))

        ret,frame=self.vcap.read()
        
        if int(cv2.__version__.split(".")[0])>2: 
            self.n_frames = int(self.vcap.get(cv2.CAP_PROP_FRAME_COUNT))-31
        else:
            self.n_frames = int(self.vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))-31
        self.n_frames=53760
        print(self.n_frames)
        
        if ret:
            if self.mode=="temporal":
                self.frame_shape=frame.shape[:2]
                self.batch_x_shape=(self.batch_size,self.frames_per_sample,self.frame_shape[0],self.frame_shape[1])
                self.batch_y_shape=(self.batch_size,self.frames_per_sample,1)
        
            elif self.mode=="stacked":
                self.frame_shape=frame.shape[:2]
                self.batch_x_shape=(self.batch_size,self.frames_per_sample*self.frame_shape[0],self.frame_shape[1])
                self.batch_y_shape=(self.batch_size,self.frames_per_sample,1)
        
            elif self.mode=="optical_flow_dense":
                self.frame_shape=frame.shape
                self.batch_x_shape=((self.batch_size,)+self.frame_shape)
                self.batch_y_shape=(self.batch_size,1)
        

            self.vcap.set(cv2.CAP_PROP_POS_FRAMES,0) # discrard that read from buffer start at the begining of the video
            print(self.frame_shape)
            print(self.batch_x_shape)
            print(self.batch_y_shape)
        else:
            raise IOError("Can't read Video Can be incorrect video codecs")

        txt_file=open(self.txt,"r")
        self.spds=txt_file.read().split("\n")
        txt_file.close()


    def build_video_folder(self):
        '''
        build video to frame idx folder
        because opencv random acess is super fucking slow (18 seconds on 32*3 frames)
        '''
        if not os.path.exists(self.db_name):
            os.mkdir(self.db_name)
            print("making dir:,",self.db_name)
        print("building video Database")
        for idx in tqdm.trange(0,self.n_frames):
            ret,frame=self.vcap.read()
            if not ret:
                break
            cv2.imwrite("{1}/{0}.png".format(idx,self.db_name),frame)
        

    def opticalFlowDense(self,image_current, image_next):
        #taken from :https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4
        """
        input: image_current, image_next (RGB images)
        calculates optical flow magnitude and angle and places it into HSV image
        * Set the saturation to the saturation value of image_next
        * Set the hue to the angles returned from computing the flow params
        * set the value to the magnitude returned from computing the flow params
        * Convert from HSV to RGB and return RGB image with same size as original image
        """
    
   
        image_current=image_current.astype('float32')
        image_next=image_next.astype('float32')
        gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)


        hsv = np.zeros((self.frame_shape))
        # set saturation
        hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
    
        # Flow Parameters
        #flow_mat = cv2.CV_32FC2
        flow_mat = None
        image_scale = 0.5
        nb_images = 1
        win_size = 15
        nb_iterations = 2
        deg_expansion = 5
        STD = 1.3
        extra = 0# obtain dense optical flow paramters
        flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                            flow_mat, 
                                            image_scale, 
                                            nb_images, 
                                            win_size, 
                                            nb_iterations, 
                                            deg_expansion, 
                                            STD, 
                                            0)


        # convert from cartesian to polar
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  

        # hue corresponds to direction
        hsv[:,:,0] = ang * (180/ np.pi / 2)

        # value corresponds to magnitude
        hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)

        # convert HSV to int32's
        hsv = np.asarray(hsv, dtype= np.float32)
        rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return rgb_flow

    def get_indexes(self):
        all_of_them_idx=np.arange(self.n_frames)
        #reshape them into sequential chunks with the temporal axes
        '''
        in frames how close can samples be from train and valid and test. for example:
        consider the idx [1,2,3]
        train has idx [1,2,3]
        if test has idxes [4,5,6] this might the make model overfit since closer frames are quite similar in x andy y values 
        so the solution is to keep them apart by distance_factor 
        solves this problem somewhat
        '''
        chunk_size=self.n_frames//self.distance_factor
        all_of_them_idx=all_of_them_idx.reshape(chunk_size,self.distance_factor)
        np.random.shuffle(all_of_them_idx)
        all_of_them_idx=all_of_them_idx.reshape(self.n_frames//self.frames_per_sample,self.frames_per_sample)
        
        idx_train=all_of_them_idx[:self.n_frames_train//self.frames_per_sample]
        idx_test=all_of_them_idx[:self.n_frames_test//self.frames_per_sample]
        idx_valid=all_of_them_idx[:self.n_frames_valid//self.frames_per_sample]
        
        print("train_samples: {0} \ntest_samples:{1} \nvalid_samples: {2}".format(idx_train.shape[0],idx_test.shape[0],idx_valid.shape[0]))

        self.vcap.release()

        return idx_train,idx_test,idx_valid

    def __len__(self):
        return self.frame_idx.shape[0]//self.batch_size

    def __getitem__(self, idx):
        if np.all(self.frame_idx)==None:
            raise ValueError("you must supply frame_idx before using the generator. refer to the docstring")
        indexes = self.frame_idx[idx*self.batch_size:(idx+1)*self.batch_size]
      
        batch_x=np.zeros(shape=self.batch_x_shape) 
        batch_y=np.zeros(shape=self.batch_y_shape) 
        
        if self.mode=="stacked" or self.mode=="optical_flow_dense":
            frames=np.zeros(shape=(self.frames_per_sample,)+self.frame_shape)

        sample=0
        sample_in_frame=0
        for idx in indexes:
            for frame_idx in idx:
                txt_i=frame_idx

                frame=cv2.imread("{1}/{0}.png".format(frame_idx,self.db_name))

                if self.mode=="stacked" or self.mode=="temporal": 
                    batch_y[sample][sample_in_frame]=self.spds[txt_i]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.mode=="temporal":
                    batch_x[sample][sample_in_frame]=frame
                
                if self.mode=="stacked" or self.mode=="optical_flow_dense":
                    frames[sample_in_frame]=frame
                

                sample_in_frame+=1
            
            if self.mode=="stacked":
                #this could be optimized by calling reshape() on the entire batch_x not on each frame of samples
                batch_x[sample]=frames.reshape(self.frame_shape[0]*self.frames_per_sample,self.frame_shape[1])
           
            if self.mode=="optical_flow_dense":
                batch_x[sample]=self.opticalFlowDense(frames[0],frames[1])
                batch_y[sample]=np.mean([float(self.spds[txt_i-1]),float(self.spds[txt_i])])
            

            sample+=1
            sample_in_frame=0

        if self.mode=="stacked" or self.mode=="temporal":
            return batch_x.reshape(self.batch_x_shape+(1,)),batch_y

        if self.mode=="optical_flow_dense":
            return batch_x,batch_y
            
if __name__=="__main__":
    import time
    
    com_gen=commai_speed_chl_gen(n_frames_valid=1000,n_frames_test=1400,n_frames_train=18000,batch_size=32)
    train_idx,test_idx,valid_idx=com_gen.get_indexes()
    t=time.time()
    com_gen.build_video_folder()
    print(time.time()-t)
    exit()
    print(train_idx)
    test_gen=commai_speed_chl_gen(frame_idx=test_idx)
    test_frames=[]
    t=time.time()
    for i in test_gen:
        test_frames.append(i)
        break
    print("took to extract 32 samples:",time.time()-t)
    print(test_frames[0][0].shape)


def extractFeatures(img):
  # Rightfuly Kindnaped from @geohots twitchslam 
  # detection
  pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
  kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
  kps, des = orb.compute(img, kps)
  return kps,des
  
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def get_orb_features_hardCoded(frame0,frame1,frame2,p_val=200):
    kp0=extractFeatures(frame0)
    kp1=extractFeatures(frame1)
    kp2=extractFeatures(frame2)

    matches_0to1 = bf.match(kp0[1],kp1[1])
    matches_1to2 = bf.match(kp1[1],kp2[1])

    matches_0to1 = sorted(matches_0to1, key = lambda x:x.distance)
    matches_1to2 = sorted(matches_1to2, key = lambda x:x.distance)


    fpts=[]
    pt_fs_0to1_0=[(int(kp0[0][i.queryIdx].pt[0]) ,int(kp0[0][i.queryIdx].pt[1])) for i in matches_0to1]
    pt_fs_0to1_1=[(int(kp1[0][i.trainIdx].pt[0]),int(kp1[0][i.trainIdx].pt[1])) for i in matches_0to1]

    pt_fs_1to2_1=[(int(kp1[0][i.queryIdx].pt[0]) ,int(kp1[0][i.queryIdx].pt[1])) for i in matches_1to2]
    pt_fs_1to2_2=[(int(kp2[0][i.trainIdx].pt[0]),int(kp2[0][i.trainIdx].pt[1])) for i in matches_1to2]

    for f_idx in range(min([len(pt_fs_0to1_0),len(pt_fs_1to2_1)])): #just loop till you reach the max number of features in the smallest match list
        pt0=pt_fs_0to1_0[f_idx]
        pt1=pt_fs_0to1_1[f_idx]

        try:
            idx_pt2=pt_fs_1to2_1.index(pt1)
        except ValueError: # means the point of the list move on
            sk+=1
            continue

        pt2=pt_fs_1to2_2[idx_pt2]

        fpts.append([f_idx,pt0,pt1,pt2])
    if len(fpts)>p_val:
        return fpts[:p_val-1]
    if len(fpts)<p_val:
        return fpts+[i for i in range((p_val-1)-len(fpts))]
    return fpts
