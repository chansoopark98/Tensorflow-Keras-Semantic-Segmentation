import numpy as np
import cv2
from imageio import imread

# img = cv2.imread('./test.jpeg')
# img = cv2.imread('./lep_image.png')
# img = cv2.imread('./sample.jpg')
img = cv2.imread('./test_img/t6.jpg')
# t2에 팽창연산 안하면 끊기는 문제 발생

img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

original = img.copy()
mask = np.zeros(img.shape).astype(img.dtype)

# 250, 200 to 400, 300
mask[200:300, 250:400]= img[200:300, 250:400]
img = mask
cv2.imshow('mask test', img)
cv2.waitKey(0)
# img = imread('./lep_image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (5, 5), 0)

img = cv2.medianBlur(img, 5)
# img = cv2.medianBlur(img, 31)
# cv2.imshow('blur', img)
# cv2.waitKey(0)

# img = np.array([
#     [6, 2, 6],
#     [7, 3, 2],
#     [5, 8, 1]
# ]



def calc_threshold(img):
    """
    라플라스 연산자를 통한 임계값 계산
    input : gray scale image (shape: [h,w])
    output : threshold
    """
    # zero_img = np.zeros(img.shape, dtype=np.float)
    # pad_img = np.pad(img, ((1,1),(1,1)), 'constant', constant_values=0).astype(float)
    # print('pad_img', pad_img)
    # for i in range(img.shape[0]):
    #     i += 1
    #     for j in range(img.shape[1]):
    #         j +=1
    #         zero_img[i-1, j-1] = (pad_img[i+1,j] + pad_img[i-1,j] + pad_img[i,j+1] + pad_img[i,j-1]) - (4 * pad_img[i,j])
    

    # laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
    cv2.imshow('laplacian', laplacian)
    cv2.waitKey(0)

    canny = cv2.Canny(img, 25, 255)
    cv2.imshow('canny', canny)
    cv2.waitKey(0)

    kernel = np.ones((1,5), np.uint8)  # note this is a horizontal kernel
    canny = cv2.dilate(canny, kernel, iterations=1)
    canny = cv2.erode(canny, kernel, iterations=1)

    cv2.imshow('after canny', canny)
    cv2.waitKey(0)
    

    print('max', np.max(laplacian))
    print('min', np.min(laplacian))
    threshold = laplacian.sum() / (img.shape[0] * img.shape[1])


    return threshold, canny

def calc_lep(img, threshold):
    pad_img = np.pad(img, ((1,1),(1,1)), 'constant', constant_values=0).astype(float)
    out_img = np.zeros(img.shape, dtype=np.float)
    
    for i in range(img.shape[0]):
        i += 1
        
        for j in range(img.shape[1]):
            j += 1
            n = 0
            if abs(pad_img[i-1, j-1]- pad_img[i, j]) <= threshold:
                n += 1
            if abs(pad_img[i-1, j]- pad_img[i, j]) <= threshold:
                n += 1
            if abs(pad_img[i-1, j+1]- pad_img[i, j]) <= threshold:
                n += 1
            if abs(pad_img[i, j+1]- pad_img[i, j]) <= threshold:
                n += 1
            if abs(pad_img[i+1, j+1]- pad_img[i, j]) <= threshold:
                n += 1
            if abs(pad_img[i+1, j]- pad_img[i, j]) <= threshold:
                n += 1
            if abs(pad_img[i+1, j-1]- pad_img[i, j]) <= threshold:
                n += 1
            if abs(pad_img[i, j-1]- pad_img[i, j]) <= threshold:
                n += 1
        
            lep = np.power(2, n) - 1

            if lep == 255:
                out_img[i-1, j-1] = 255
            
            if i <= 20 or j <= 20:
                out_img[i-1, j-1] = 0
                # pad_img[i, j] = 255
    
    return out_img

        
threshold, canny = calc_threshold(img)
print(threshold)
output = calc_lep(canny, 127)
output = canny.astype(np.uint8)

cv2.imshow('output', output)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img = np.where(output >= 255, img, 0)
cv2.imshow('img', img)
cv2.waitKey(0)

        
# contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(canny, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


contour_list = []
i = 0
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)
    # print('area', area)
    # if area < 500:
    #     print(i)
        
    contour_list.append(contour)
    i+=1

msg = "Total holes: {}".format(len(approx)//2)

cv2.putText(original, msg, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
 
mask = np.zeros(original.shape).astype(original.dtype)

for idx in range(len(contour_list)):
    draw_mask = mask.copy()
    cv2.drawContours(draw_mask, contour_list, idx, (255, 255, 255), -1)
    cv2.imshow('Objects Detected', draw_mask)
    cv2.waitKey(0)

    # center, radius = cv2.minEnclosingCircle(contour_list[-1])
    # print(center)
    
    # cv2.circle(original, (int(center[0]), int(center[1])), int(radius), (255,0,0), -1)
    
# stencil = np.zeros(original.shape).astype(original.dtype)
# cv2.fillPoly(stencil, contours, (255,255,255))
# result = cv2.bitwise_and(original, stencil)
# cv2.imshow('result', result)
# cv2.waitKey(0)


