## ROBT 310 - Final Project - Virtual Keyboard
## Team: Daryn Kalym, Alibek Manabayev, Alexey Muryshkin
## Date: April 27, 2018

## modules
import numpy as np
from math import sqrt
import cv2
import sys

from KeyboardLayout import identify_keyboard, transform_image
from transform import four_point_transform

## global variables
sep_dist = 10 # cm
focal_len = 0.367 #cm

#frame1, frame2 = None, None
#background1, background2 = None, None
#col_index_matrixY1, col_index_matrixY2 = None, None
#col_index_matrixX1, col_index_matrixX2 = None, None

## functions

# Useless Function
def q(*x, **y):
    pass

def makeDisparity(imgL, imgR):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    return disparity

# Noise Filtering
def noiseFiltering(frame):
    res = np.uint8( frame )
    res = grayImage = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res = cv2.equalizeHist(res)
    res = medFiltImage = cv2.medianBlur(res, 3)
    
    return res

def processFrame(frame, background):    
    threshold = 50
    
    res = np.abs(np.int32(frame) - np.int32(background))
    
    mask = np.abs( np.int32(frame) - np.int32(background) ) > threshold
    backSubImage = np.zeros(frame.shape)
    backSubImage[mask] = 255
    res = backSubImage
    
    kernel = np.ones((7,7),np.uint8)
    erosion = cv2.erode(backSubImage, kernel, iterations = 1)
    kernel = np.ones((10,10),np.uint8)
    res = dilation = cv2.dilate(erosion, kernel, iterations = 1)
    
    return np.uint8( res )

def getCenterY(img, col_index_matrix):    
    mask = img == 255
    #n = np.sum(np.uint32(mask))
    #ySum = np.sum( col_index_matrix[mask] )
    
    #return int( round(ySum / n) ) if n else None
    return np.min( col_index_matrix[mask] ) if mask.any() else None

def getCenterX(img, col_index_matrix):    
    mask = img == 255
    #n = np.sum(np.uint32(mask))
    #xSum = np.sum( col_index_matrix[mask] )
    
    #return int( round(xSum / n) ) if n else None
    return np.min( col_index_matrix[mask] ) if mask.any() else None
    
def displayVideoRealTime(cap1, cap2):
    global sep_dist, focal_len
    
    # background reading
    ret, background1 = cap1.read()
    #background1 = noiseFiltering( np.flip( np.array(background1, dtype=np.uint8), axis = 1 ) )
    layout1, points1, background1 = identify_keyboard(background1)

    ret, background2 = cap2.read()
    #background2 = noiseFiltering( np.flip( np.array(background2, dtype=np.uint8), axis = 1 ) )
    layout2, points2, background2 = identify_keyboard(background2)
            
    if points1 is not None and points2 is not None and len(points1) == 4 and len(points2) == 4: 
        background1 = transform_image(background1, points1)
        background2 = transform_image(background2, points2)
        
        background1 = noiseFiltering( np.flip( np.array(background1, dtype=np.uint8), axis = 1 ) )
        background2 = noiseFiltering( np.flip( np.array(background2, dtype=np.uint8), axis = 1 ) )        
        
        new_shape = tuple(np.minimum(background1.shape, background2.shape))
        background1 = cv2.resize(background1, new_shape[::-1])
        background2 = cv2.resize(background2, new_shape[::-1])    
    
    col_index_matrixY1 = np.array( [[j for j in range(background1.shape[1])] for i in range(background1.shape[0])] )
    col_index_matrixX1 = np.array( [[i for j in range(background1.shape[1])] for i in range(background1.shape[0])] )    
    col_index_matrixY2 = np.array( [[j for j in range(background2.shape[1])] for i in range(background2.shape[0])] )
    col_index_matrixX2 = np.array( [[i for j in range(background2.shape[1])] for i in range(background2.shape[0])] )
    
    # creating video streaming windows
    o_window_name1 = "WebCam Video Streaming 1"
    #cv2.namedWindow(o_window_name1, flags=cv2.WINDOW_AUTOSIZE)
    #cv2.moveWindow(o_window_name1, 0, 0)
    
    p_window_name1 = "Processed Video Streaming 1"
    cv2.namedWindow(p_window_name1, flags=cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(p_window_name1, 0, 0)
    
    b_window_name1 = "Background Image 1"
    cv2.namedWindow(b_window_name1, flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(b_window_name1, background1.shape[1] // 3, background1.shape[0] // 3)
    cv2.moveWindow(b_window_name1, 0, 0)
    cv2.imshow(b_window_name1, np.uint8( background1 ))
    
    o_window_name2 = "WebCam Video Streaming 2"
    cv2.namedWindow(o_window_name2, flags=cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(o_window_name2, background2.shape[1], 0)
    
    p_window_name2 = "Processed Video Streaming 2"
    cv2.namedWindow(p_window_name2, flags=cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(p_window_name2, background2.shape[1], 0)
    
    b_window_name2 = "Background Image 2"
    cv2.namedWindow(b_window_name2, flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow(b_window_name2, background2.shape[1] // 3, background2.shape[0] // 3)
    cv2.moveWindow(b_window_name2, background2.shape[1], 0)
    cv2.imshow(b_window_name2, np.uint8( background2 ))
    
    while True:        
        ret, frame1 = cap1.read()
        if not ret:
            break
            
        ret, frame2 = cap2.read()
        if not ret:
            break
        
        key = cv2.waitKey(30)
        c = chr(key & 255)
        if c in ['q', 'Q', chr(27)]:
            break
        elif c in ['b', 'B']:
            background1 = np.copy(frame1)
            layout1, points1, background1 = identify_keyboard(background1)
            
            background2 = np.copy(frame2)
            layout2, points2, background2 = identify_keyboard(background2)
            
            if points1 is not None and points2 is not None and len(points1) == 4 and len(points2) == 4: 
                background1 = transform_image(background1, points1)
                background2 = transform_image(background2, points2)   
                
                background2 = noiseFiltering( np.flip( np.array(background2, dtype=np.uint8), axis = 1 ) )
                background1 = noiseFiltering( np.flip( np.array(background1, dtype=np.uint8), axis = 1 ) )                
                
                new_shape = tuple(np.minimum(background1.shape, background2.shape))
                background1 = cv2.resize(background1, new_shape[::-1])
                background2 = cv2.resize(background2, new_shape[::-1])            
            
            cv2.resizeWindow(b_window_name1, background1.shape[1] // 3, background1.shape[0] // 3)
            cv2.moveWindow(b_window_name1, 0, 0)   
            
            cv2.resizeWindow(b_window_name2, background2.shape[1] // 3, background2.shape[0] // 3)
            cv2.moveWindow(b_window_name2, background2.shape[1], 0)            
            
            cv2.imshow(b_window_name1, np.uint8( background1 ))
            cv2.imshow(b_window_name2, np.uint8( background2 ))   
            
            col_index_matrixY1 = np.array( [[j for j in range(background1.shape[1])] for i in range(background1.shape[0])] )
            col_index_matrixX1 = np.array( [[i for j in range(background1.shape[1])] for i in range(background1.shape[0])] )   
            
            col_index_matrixY2 = np.array( [[j for j in range(background2.shape[1])] for i in range(background2.shape[0])] )
            col_index_matrixX2 = np.array( [[i for j in range(background2.shape[1])] for i in range(background2.shape[0])] )            
                    
        proc_img1 = np.copy(frame1)
        proc_img2 = np.copy(frame2)        
                    
        if points1 is not None and points2 is not None and len(points1) == 4 and len(points2) == 4:
            frame1 = transform_image(frame1, points1)
            frame2 = transform_image(frame2, points2)  
            
            frame1 = cv2.resize(frame1, new_shape[::-1])
            frame2 = cv2.resize(frame2, new_shape[::-1])   
            
            frame1 = np.flip( np.array(frame1, dtype=np.uint8), axis = 1 )
            frame2 = np.flip( np.array(frame2, dtype=np.uint8), axis = 1 )                  
            
            proc_img1 = np.copy(frame1)
            proc_img2 = np.copy(frame2)
            
            frame1 = cv2.cvtColor( frame1 , cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor( frame2 , cv2.COLOR_BGR2GRAY)              
            
            cv2.imshow(p_window_name1, np.uint8( frame1 ))
            cv2.imshow(p_window_name2, np.uint8( frame2 ))              
            
            proc_img1 = noiseFiltering( proc_img1 )
            proc_img2 = noiseFiltering( proc_img2 )            
            
            proc_img1 = processFrame(proc_img1, background1)
            proc_img2 = processFrame(proc_img2, background2)
            ##cv2.imshow(p_window_name, np.uint8( proc_img ))
            
            centerY1 = getCenterY(proc_img1, col_index_matrixY1)
            centerY2 = getCenterY(proc_img2, col_index_matrixY2)
            centerX1 = getCenterX(proc_img1, col_index_matrixX1)
            centerX2 = getCenterX(proc_img2, col_index_matrixX2)
                
            if centerX1 is not None and centerX2 is not None:
                mask = proc_img1 != 255
                stereo = makeDisparity(cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE))
                #stereo[mask] = 0
                #stereo_dif = np.abs(np.int32( makeDisparity(frame1, frame2) ) - np.int32( makeDisparity(background1, background2) ) )
                #print(stereo_dif)
                cv2.imshow(o_window_name2, stereo) 
            
            if centerY2 is not None:
                proc_img2[:, centerY2] = 255
                
            if centerY1 is not None:
                proc_img1[:, centerY1] = 255
                
            if centerX2 is not None:
                proc_img2[centerX2, :] = 255
                
            if centerX1 is not None:
                proc_img1[centerX1, :] = 255            
            #if centerY1 is not None and centerY2 is not None:
                #pass
                ##print(sep_dist * focal_len / abs(centerY1 - centerY2) if abs(centerY1 - centerY2) > 0 else 0)            
            
        #cv2.imshow(p_window_name1, np.uint8( proc_img1 ))
        #cv2.imshow(p_window_name2, np.uint8( proc_img2 ))        
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    

def getVideoCapture(idd):
    if isinstance(idd, int):
        cap = cv2.VideoCapture( idd )
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('Cannot initialize video capture with id {}'.format(idd))
        #sys.exit(-1)
        
    return cap

    
## main program
def main():
    devId1 = int( input('Please enter the id of the opened video capturing device #1:\n') )
    cap1 = getVideoCapture(devId1)
    devId2 = int( input('Please enter the id of the opened video capturing device #2:\n') )
    cap2 = getVideoCapture(devId2)
    
    displayVideoRealTime(cap1, cap2)

    
if __name__ == '__main__':
    main()
