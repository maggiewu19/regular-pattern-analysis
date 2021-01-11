import cv2 

def sharpen(image):
    image = image.copy()
    gaussian = cv2.GaussianBlur(image, (0, 0), 5.0)
    image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0, image)  

    return image 

def deblur(image): 
    image = image.copy() 
    image = sharpen(sharpen(image))

    return image 