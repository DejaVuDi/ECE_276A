'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
# =============================================================================
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal
# =============================================================================
import numpy as np

class BarrelDetector():
    def __init__(self):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        self.colors = ['blue','brown','gray','green','red','black','tree','lightblue']
        self.mu = {'blue':np.array([105.66262518, 193.92050213, 130.24641307]),'brown':np.array([ 11.18391257,  88.64106566, 195.63316518]),'gray':np.array([ 25.93880084,  68.46110728, 116.20915129] ),'green':np.array([ 29.04732647, 117.04304746,  95.97473531] ),'red':np.array([170.87783451, 163.66121479, 204.7034507 ] ),'black':np.array([74.45709244, 57.610345,   41.94487932] ),'tree':np.array([ 90.48489756,   8.00378863, 179.55863382]),'lightblue':np.array( [ 86.63224897,  84.8091826,  142.13220671])}
        self.sigma = {'blue':np.array([[ 174.29762165,  128.9281938,   146.17869757], [ 128.9281938,  2530.87135811, 1280.39755825], [ 146.17869757, 1280.39755825, 3032.2442387 ]]),'brown':np.array([[  43.13325385,  -26.94444308,   27.05697898], [ -26.94444308,  421.90787397,  -54.97619334], [  27.05697898,  -54.97619334, 1151.63296129]]),'gray':np.array([[  159.45267892,   324.64861319,  -349.38484897], [  324.64861319,  1623.90145569, -1600.18179204], [ -349.38484897, -1600.18179204,  1758.95198199]]),'green':np.array([[  54.23601398,   80.02236242,  -64.81604278], [  80.02236242, 1211.86215923, -662.5804797 ], [ -64.81604278, -662.5804797,  1094.33304528]]),'red':np.array([[ 1163.06426686,   733.29072993,  -269.10802577], [  733.29072993,  3335.63784144, -1795.72683495], [ -269.10802577, -1795.72683495,  1372.58076592]]),'black':np.array( [[1379.74736862,  449.1048299,   713.53288621], [ 449.1048299,   987.61732059,   39.02006487], [ 713.53288621,   39.02006487, 1191.41133763]]),'tree':np.array([[1096.22196546,   83.7970888,  -767.57843735], [  83.7970888,    32.2825824,  -207.32889857], [-767.57843735, -207.32889857, 3585.90480761]]),'lightblue':np.array( [[1064.91683129,  -74.29573639,   10.79119341], [ -74.29573639, 1193.52190228, -217.80482674], [  10.79119341, -217.80482674,  673.37997143]])}
        prior = [0.18337287, 0.10223319, 0.18497515, 0.10710646, 0.02567254, 0.18315186, 0.04771973, 0.1657682 ]
        self.Gu = [self.Gussian(color) for color in self.colors]
        self.value = lambda x: np.argmax([self.Gu[i](x)*prior[i] for i in range(len(self.colors))],axis=0)
        self.pdf = lambda x,mean,cov: (1/np.sqrt((2*np.pi) ** 3 * np.linalg.det(cov)))*np.exp(-0.5*((x-mean)@np.linalg.inv(cov)*(x-mean)).sum(1))
        #raise NotImplementedError

    def Gussian(self,color):
        return lambda x:self.pdf(x, self.mu[color],self.sigma[color])
# =============================================================================
#     def Gussian(self,color):
#         return lambda x: multivariate_normal.pdf(x, mean=self.mus[color], cov=self.sigmas[color])
# =============================================================================

    def segment_image(self, img):
        '''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
        [m,n,_] = img.shape
        test_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(m*n,3)
        pred = self.value(test_img)
        pred[pred==0] = 255
        pred[pred<255] = 0
        pred[pred==255] = 1
        mask_img = pred.reshape(m,n)
        #cv2.imwrite('seg/'+filename,mask_img)
        #raise NotImplementedError
        return mask_img
        
    def get_bounding_box(self, img):
        '''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
        [m,n,_] = img.shape
# =============================================================================
#         mask = self.segment_image(img)*255
#         mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
#         contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#         transformed = cv2.drawContours(mask,contours,-1,(0,255,0),3)
#         transformed = np.stack((transformed,)*3, axis=-1)
# =============================================================================
        kernelc = np.ones((20,30),np.uint8) 
        kernelo = np.ones((10,10),np.uint8) 
        closing = lambda img: cv2.morphologyEx(np.uint8(img),cv2.MORPH_CLOSE, kernelc) 
        opening = lambda img: cv2.morphologyEx(np.uint8(img),cv2.MORPH_OPEN, kernelo)     
        transformed = closing(opening(self.segment_image(img)*255))
        
        label_img = label(transformed)
        #cv2.imwrite('morph/'+filename,transformed)
        regions = regionprops(label_img,coordinates='xy') 
        boxes=[]
        for props in regions: 
            #props.area>500 and props.area<23000 and
            ori = props.orientation
            length1 = props.minor_axis_length
            length2 = props.major_axis_length       
            if  (ori<-0.5 or ori>0.5) and length1/length2<0.67 and length1/length2>0.4: 
                y1,x1,y2,x2 = props.bbox
                boxes.append([x1,y1,x2,y2])
                #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                #cv2.imwrite('bbox/'+filename,img)
		#raise NotImplementedError
        #print(boxes)
        #txt_file.write(filename+boxes)
        return boxes


if __name__ == '__main__':
    folder = "trainset"
    #txt_file = open("output.txt","w")
    my_detector = BarrelDetector()
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imwrite('seg/'+filename,my_detector.segment_image(img)*255)
        print(my_detector.get_bounding_box(img))
        
		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

