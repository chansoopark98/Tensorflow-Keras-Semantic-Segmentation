import tensorflow as tf
import cv2
import numpy as np

def canny_selector(rgb):

    cv2.namedWindow("Data augmentation sample test")
    cv2.createTrackbar("saturation", "Data augmentation sample test", 2, 10, lambda x : x)
    cv2.createTrackbar("brightness", "Data augmentation sample test", 1, 10, lambda x : x)
    cv2.createTrackbar("contrast", "Data augmentation sample test", 2, 10, lambda x : x)
    

    cv2.setTrackbarPos("saturation", "Data augmentation sample test", 2)
    cv2.setTrackbarPos("brightness", "Data augmentation sample test", 1)
    cv2.setTrackbarPos("contrast", "Data augmentation sample test", 2)
    


    
    while cv2.waitKey(1) != ord('q'):
        img = rgb.copy()
        saturation = cv2.getTrackbarPos("saturation", "Data augmentation sample test")
        saturation *= 0.1
        brightness = cv2.getTrackbarPos("brightness", "Data augmentation sample test")
        brightness *= 0.1
        contrast = cv2.getTrackbarPos("contrast", "Data augmentation sample test")
        contrast *= 0.1

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = tf.image.random_saturation(img, 0.1, saturation)
    
        # img = tf.image.random_brightness(img, brightness)
    
        img = tf.image.random_contrast(img, 0.1, contrast)
        
        img = img.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.moveWindow("Data augmentation sample test", 800, 400)
        cv2.imshow("Data augmentation sample test", img)
        cv2.waitKey(100)
        
    
    return img


if __name__ == '__main__':
    # Load Test image
    img = cv2.imread('test.jpg')
    img = cv2.resize(img, dsize=(512, 512))
    canny_selector(rgb=img)

