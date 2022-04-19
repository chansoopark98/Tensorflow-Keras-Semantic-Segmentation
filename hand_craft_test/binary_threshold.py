import argparse
import time
import os
import glob
import cv2
import numpy as np
from data_labeling.utils import canny_selector, find_contours

def check_boolean(value):
    if value == 1:
        return True
    else:
        return False

def fit_rotated_ellipse_ransac(data,iter=30,sample_num=10,offset=80.0):

	count_max = 0
	effective_sample = None

	for i in range(iter):
		sample = np.random.choice(len(data), sample_num, replace=False)

		xs = data[sample][:,0].reshape(-1,1)
		ys = data[sample][:,1].reshape(-1,1)

		J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
		Y = np.mat(-1*xs**2)
		P= (J.T * J).I * J.T * Y

		# fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
		a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
		ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

		# threshold 
		ran_sample = np.array([[x,y] for (x,y) in data if np.abs(ellipse_model(x,y)) < offset ])

		if(len(ran_sample) > count_max):
			count_max = len(ran_sample) 
			effective_sample = ran_sample

	return fit_rotated_ellipse(effective_sample)


def fit_rotated_ellipse(data):

	xs = data[:,0].reshape(-1,1) 
	ys = data[:,1].reshape(-1,1)

	J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
	Y = np.mat(-1*xs**2)
	P= (J.T * J).I * J.T * Y

	a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
	theta = 0.5* np.arctan(b/(a-c))  
	
	cx = (2*c*d - b*e)/(b**2-4*a*c)
	cy = (2*a*e - b*d)/(b**2-4*a*c)

	cu = a*cx**2 + b*cx*cy + c*cy**2 -f
	w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
	h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))

	ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

	error_sum = np.sum([ellipse_model(x,y) for x,y in data])
	print('fitting error = %.3f' % (error_sum))

	return (cx,cy,w,h,theta)

color_list = [(238,0,0),(0,252,124),(142,56,142),(10,20,0),(245,245,245)]

parser = argparse.ArgumentParser()
parser.add_argument("--result_path",     type=str,   help="test result path", default='./test_imgs/hand/')

args = parser.parse_args()
RESULT_PATH = args.result_path

batch_idx = 0
avg_duration = 0
ng_time = 0

img_list = glob.glob(os.path.join(RESULT_PATH,'*.jpg'))
img_list.sort()

def onChange(pos):
    pass

for i in range(len(img_list)):
    
    rgb = cv2.imread(img_list[i])
    gray = cv2.cvtColor(rgb.copy(), cv2.COLOR_BGR2GRAY)


    
    _, binary = cv2.threshold(gray.copy(), 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circle_contour = []
    if len(contours) != 0:
        tmp_area = 0
        idx = -1
        for contour in contours:
            area = cv2.contourArea(contour)
            if tmp_area <= area:
                tmp_area = area
                idx += 1

                circle_contour.append(contour)
            
        # if len(circle_contour) != 0:
        #     x,y,w,h = cv2.boundingRect(circle_contour[0])

    cv2.drawContours(binary, circle_contour, idx, 127, -1)
    # draw_area = draw_img.copy()
    # binary = np.expand_dims(np.where(binary == 127, 1, 0).astype(np.uint8), -1)
    binary = np.where(binary == 127, 1, 0).astype(np.uint8)

    draw_gray = gray.copy() * binary



    # circles = cv2.HoughCircles(draw_gray, cv2.HOUGH_GRADIENT, 1, minDistance,
    #         param1=CannyThreshold, param2=CenterThreshold, minRadius=minRadius, maxRadius=maxRadius)
    # if circles is not None:
    #     for i in circles[0]:
    #         cv2.circle(draw_gray, (int(i[0]), int(i[1])), int(i[2]), 0, 1)
    #         cv2.circle(draw_gray, (int(i[0]), int(i[1])), 0, 0, -1)

    
    # draw_gray = cv2.cvtColor(draw_gray, cv2.COLOR_BGR2HSV)
    # draw_gray = draw_gray[:,:,1]
    draw_gray = cv2.GaussianBlur(draw_gray, (7, 7), 0.5)
    mask = canny_selector(draw_gray)
    
    rgb_result = find_contours(mask, rgb.copy())
        
        
            

print(f"avg inference time : {(avg_duration / len(img_list)) // 1000000}ms.")
print(f"No good images : {ng_time}.")