import cv2

def find_and_draw_contours(img, original_mask):
     # Get display area
    contours, hier = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    

    # 외곽선 그리기
    # hier 계층 정보를 입력했으므로 모든 외곽선에 그림을 그리게 됩니다.
    # for idx in range(len(display_contours)):
    #     img = cv2.drawContours(img, display_contours, idx, (127, 127, 127), -1, cv2.LINE_8, hier)
    #     img = cv2.drawContours(img, display_contours, idx, (255, 255, 255), 2, cv2.LINE_8, hier)

    
    img = cv2.drawContours(img, display_contours, 0, (255, 0, 0), -1, cv2.LINE_8, hier)
    img = cv2.drawContours(img, display_contours, 0, (255, 255, 255), 2, cv2.LINE_8, hier)
    print(len(display_contours))
    if len(display_contours) >= 2:
        img = cv2.drawContours(img, display_contours, 1, (0, 0, 255), -1, cv2.LINE_8, hier)
        img = cv2.drawContours(img, display_contours, 1, (255, 255, 255), 2, cv2.LINE_8, hier)

        
    return img


