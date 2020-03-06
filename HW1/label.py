"""
Roipoly
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
https://github.com/jdoepfert/roipoly.py
@author Di Gu
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import sys

def handle(img,mask):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # Convert BGR to HSV
    [m,n,_] = img.shape
    res = cv2.bitwise_and(img,img,mask = np.uint8(mask)) # Bitwise-AND mask and original image
    data = res.reshape(m*n,3)
    data = np.array(list(filter(lambda x: (x>=1).any(),data))) # Filter the black
    return data

def label():
    if len(sys.argv)>=2:
        color = sys.argv[1]
    else:
        print('color?')
        color = input()
    ni = ''
    
    data = np.array([0,0,0])

    while(not ni == 'q'):
        print("input image number:")
        ni = input()
        if ni =='q':
            break
        else:
            fig = plt.figure()
            img = cv2.imread('trainset/'+ni+'.png')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),interpolation='nearest')
            roi = RoiPoly(color='r',fig=fig)
            mask = roi.get_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            data = np.vstack((data,handle(img,mask))) # Stack data sample
            data = np.array(list(filter(lambda x: (x>=1).any(),data)))
            print(str(data.shape))
            print("label next press ENTER, finish label press Q then ENTER")
            ni = input()
        np.save(color+'.npy',data)
        
if __name__=='__main__':
    label()