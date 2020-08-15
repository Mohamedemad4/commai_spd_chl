import cv2
import sys

import pygame
from pygame.locals import *
import cv2
import numpy as np
import time
#This shows an image weirdly...
screen_width, screen_height = 640, 480
screen=pygame.display.set_mode((screen_width,screen_height))

c=cv2.VideoCapture(sys.argv[1])
txt_file=open(sys.argv[2],'r').read().split("\n")
txt_idx=0



def getCamFrame():
    global txt_idx
    ret,frame=c.read()
    frame=frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cv2.putText(frame, txt_file[txt_idx][::-1], (0,70),0,3,255)
    
    frame=np.rot90(frame)
    frame=pygame.surfarray.make_surface(frame)
    txt_idx+=1
    
    return frame

def blitCamFrame(frame,screen):
    screen.blit(frame,(0,0))
    return screen

screen.fill(0) #set pygame screen to black
frame=getCamFrame()
screen=blitCamFrame(frame,screen)
pygame.display.flip()

running=True
while running:
    for event in pygame.event.get(): #process events since last loop cycle
        if event.type == KEYDOWN:
            running=False
    frame=getCamFrame()
    screen=blitCamFrame(frame,screen)
    pygame.display.flip()
    time.sleep(1/20)


pygame.quit()
