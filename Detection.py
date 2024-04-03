import jetson.inference    
import jetson.utils        
import cv2                 
import numpy as np         

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold = 0.5)     
camera = jetson.utils.gstCamera(0)  
while 1:
    img, width, height = camera.CaptureRGBA(zeroCopy = 1)
    detections = net.Detect(img, width, height)    
    image = jetson.utils.cudaToNumpy(img,width, height, 4)   
    image1 = cv2.cvtColor (image.astype (np.uint8), cv2.COLOR_RGBA2BGR)  
    cv2.imshow ("目标检测",image1)   

    kk = cv2.waitKey(1)   
    if kk == ord('q'):  
        break 
