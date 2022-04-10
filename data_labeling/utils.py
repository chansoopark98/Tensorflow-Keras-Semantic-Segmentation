import cv2
import numpy as np

def canny_selector(rgb):
    cv2.namedWindow("Canny Selector")
    cv2.createTrackbar("minValue", "Canny Selector", 1, 255, lambda x : x)
    cv2.createTrackbar("maxValue", "Canny Selector", 1, 255, lambda x : x)
    cv2.createTrackbar("black threshold", "Canny Selector", 1, 255, lambda x : x)
    cv2.createTrackbar("Kernel Size", "Canny Selector", 1, 7, lambda x : x)
    cv2.createTrackbar("v_dilate", "Canny Selector", 0, 1, lambda x : x)
    cv2.createTrackbar("h_dilate", "Canny Selector", 0, 1, lambda x : x)
    cv2.createTrackbar("v_erode", "Canny Selector", 0, 1, lambda x : x)
    cv2.createTrackbar("h_erode", "Canny Selector", 0, 1, lambda x : x)
    

    cv2.setTrackbarPos("minValue", "Canny Selector", 127)
    cv2.setTrackbarPos("maxValue", "Canny Selector", 255)
    cv2.setTrackbarPos("black threshold", "Canny Selector", 64)
    cv2.setTrackbarPos("Kernel Size", "Canny Selector", 3)
    cv2.setTrackbarPos("v_dilate", "Canny Selector", 1)
    cv2.setTrackbarPos("h_dilate", "Canny Selector", 1)
    cv2.setTrackbarPos("v_erode", "Canny Selector", 1)
    cv2.setTrackbarPos("h_erode", "Canny Selector", 1)
    

    img = rgb.copy()

    while cv2.waitKey(1) != ord('q'):
        # 수직 = VERTICAL
        # 수평 = horizontal
        thresh = cv2.getTrackbarPos("minValue", "Canny Selector")
        maxval = cv2.getTrackbarPos("maxValue", "Canny Selector")
        black_t = cv2.getTrackbarPos("black threshold", "Canny Selector")
        kernel_size = cv2.getTrackbarPos("Kernel Size", "Canny Selector")
        v_dilate = cv2.getTrackbarPos("v_dilate", "Canny Selector")
        h_dilate = cv2.getTrackbarPos("h_dilate", "Canny Selector")
        v_erode = cv2.getTrackbarPos("v_erode", "Canny Selector")
        h_erode = cv2.getTrackbarPos("h_erode", "Canny Selector")
        
        img = np.where(rgb.copy() <= black_t, 0, rgb.copy())
        canny = cv2.Canny(img, thresh, maxval)
        
        v_kernel = np.ones((kernel_size, 1), np.uint8) 
        h_kernel = np.ones((1,kernel_size), np.uint8) 

        if v_dilate == 1:
            canny = cv2.dilate(canny, v_kernel, iterations=1)    
        
        if h_dilate == 1:
            canny = cv2.dilate(canny, h_kernel, iterations=1)    
        
        if v_erode == 1:
            canny = cv2.erode(canny, v_kernel, iterations=1)    

        if h_erode == 1:
            canny = cv2.erode(canny, h_kernel, iterations=1)    

        cv2.imshow("Canny Selector", canny)
    
    return canny

def load_imgs(img, resize=(640, 480)):
    # img = cv2.imread('./data_labeling/test_img/t2.jpg')
    if resize:
        img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_AREA)
    output_img = img.copy()

    x, y, w, h = cv2.selectROI(img)
    
    # img = cv2.GaussianBlur(img, (5,5), 0) # 이미지에 블러 하면 홀은 잘 찾음
    # img = cv2.GaussianBlur(img, (7,7), 0) # 이미지에 블러 하면 홀은 잘 찾음
    mask = np.zeros(img.shape).astype(img.dtype)

    # 250, 200 to 400, 300
    # mask[200:300, 250:400]= img[200:300, 250:400]
    mask[y:y+h, x:x+w]= img[y:y+h, x:x+w]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    
    # mask = cv2.medianBlur(mask, 7)

    return mask, output_img

def canny_edge(img_input, use_vertical=True, use_horizontal=True):

    # canny = cv2.Canny(img, 25, 255) # hole is good
    # white white_plus = 100
    # gray white_plus = 150
    img = np.where(np.logical_and(img_input>=1, img_input<=100), img_input+100, img_input)
    img = np.clip(img, 0, 255)
    img = img.astype(img_input.dtype)
    cv2.imshow('np where', img)
    cv2.waitKey(0)
    canny = cv2.Canny(img_input, 25, 100) # hole is good

    cv2.imshow('canny', canny)
    cv2.waitKey(0)
    if use_horizontal:
        kernel = np.ones((1,5), np.uint8) 
        canny = cv2.dilate(canny, kernel, iterations=1)
        kernel = np.ones((1,3), np.uint8) 
        canny = cv2.erode(canny, kernel, iterations=1)

    if use_vertical:
        v_kernel = np.ones((5,1), np.uint8)  
        canny = cv2.dilate(canny, v_kernel, iterations=1)
        # canny = cv2.erode(canny, kernel, iterations=1)

    # canny = cv2.erode(canny, v_kernel, iterations=1)

    return canny

def find_contours(mask, img, color=(0, 255, 0)):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    mask = np.where(mask >= 255, mask, 0)

    contour_list = []
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if area >= 1000:
            
            contour_list.append(contour)

    
        

    # msg = "Total holes: {}".format(len(approx)//2)
    # cv2.putText(original, msg, (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
    
    draw_mask = []
    draw_hole = []
    zero_mask = np.zeros(img.shape)[:, :, 0].astype(img.dtype)

    for idx in range(len(contour_list)):
        draw_img = img.copy()
        cv2.drawContours(draw_img, contour_list, idx, color, -1)
        cv2.imshow('Objects Detected', draw_img)

        key = cv2.waitKey(0)
        print(key)
        cv2.destroyAllWindows()
        
        delete_idx = abs(48 - key)
        
        if delete_idx == 65:
            break
        
        try:
            # 1번 키를 누를 때
            if key == 49:
                draw_mask = zero_mask.copy()
                cv2.drawContours(draw_mask, contour_list, idx, (255, 255, 255), -1)
                
            # elif key == 50:
            #     draw_hole = zero_mask.copy()
            #     cv2.drawContours(draw_hole, contour_list, idx, (255, 255, 255), -1)
                

        except:
            print('Out of range!!')

    return draw_mask