#ALBAN DIETRICH r0877856

import argparse
import cv2
import sys
import numpy as np
import skimage.exposure as exposure

def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def write_text(frame, text, pos, size, color):
    #Write text on the frame
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color,2)

def display_conclusion(frame):
    #Dispay conclusion of difference between Gaussian and bilateral filters
    cv2.putText(frame, "Conclusion: In contrast to the Gaussian filter,", (100,90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,190,246),2)
    cv2.putText(frame, "the bilateral filter is an edge preserving filter", (100,115), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,190,246),2)
    cv2.putText(frame, "thanks to its sigma color which takes into account", (100,140), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,190,246),2)
    cv2.putText(frame, "the intensity of the pixels (the bilateral also", (100,165), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,190,246),2)
    cv2.putText(frame, "has a spatial sigma like the Gaussian filter)", (100,190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,190,246),2)

def convert_to_gray(frame):
    #Create grayscale image + creating 3 channels for the output
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_3_dim = np.zeros_like(frame)
    frame_3_dim[:,:,0] = frame_gray
    frame_3_dim[:,:,1] = frame_gray
    frame_3_dim[:,:,2] = frame_gray
    return frame_3_dim


def convert_to_displayable_frame(frame, frame_mod):
    #Create frame with 3 channels
    frame_3_dim = np.zeros_like(frame)
    frame_3_dim[:,:,0] = frame_mod
    frame_3_dim[:,:,1] = frame_mod
    frame_3_dim[:,:,2] = frame_mod
    return frame_3_dim

def create_hsv_color_edge(sobel_x, sobel_y):
    #We go in polar coordinates to have certain colors to highlight the edges 
    mag, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees = True)
    
    #Normalization
    mag = exposure.rescale_intensity(mag, in_range='image', out_range=(0,255)).clip(0,255).astype(np.uint8)
    angle = exposure.rescale_intensity(angle, in_range='image', out_range=(0,180)).clip(0,180).astype(np.uint8)
    
    hsv = cv2.merge([angle, mag, mag])
    hsv[:,:,1] = 255
    return hsv

def hough_transform_circle(frame, dp, minDist, param1, param2, minRadius, maxRadius):
    #Function to detect circular objects
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Reduce noice with median blur
    gray = cv2.medianBlur(gray, 5)
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, rows * minDist,
                               param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            #Center of the circle
            cv2.circle(frame, center, 2, (0, 0, 255), 3)
            #Circle perimeter
            radius = i[2]
            cv2.circle(frame, center, radius, (0, 0, 255), 3)

def main(input_video_file: str, output_video_file: str) -> None:
    #Objects to work with OpenCV
    cap = cv2.VideoCapture(input_video_file)

    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #Saving output into mp4 format
    isColor = True
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height), isColor)

    ret, frame = cap.read()
    
    #Tracker initialization
    tracker = cv2.TrackerCSRT_create()
    bbox = (735, 362, 70, 70)
    ret_track = tracker.init(frame, bbox)
    
    #some parameter for dispay
    size = 0.75
    color = (0,190,246) #Dark yellow
    
    #Size of smiley box
    size_smiley = 1.2
    
    #Principal loop
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 500, 1000):
                frame = convert_to_gray(frame)
                write_text(frame, "Grayscale image", (100,50), size, color)
            elif between(cap, 1500, 2000):
                frame = convert_to_gray(frame)
                write_text(frame, "Grayscale image", (100,50), size, color)
            elif between(cap, 2500, 3000):
                frame = convert_to_gray(frame)
                write_text(frame, "Grayscale image", (100,50), size, color)
            elif between(cap, 3500, 4000):
                frame = convert_to_gray(frame)
                write_text(frame, "Grayscale image", (100,50), size, color)
            elif between(cap, 4000, 5000):
                frame = cv2.GaussianBlur(frame,(15,15),1)
                write_text(frame, "Gaussian filter - kernel (15x15) - sigma 1", (100,50), size, color)
            elif between(cap, 5000, 6000):
                frame = cv2.GaussianBlur(frame,(15,15),3)
                write_text(frame, "Gaussian filter - kernel (15x15) - sigma 3", (100,50), size, color)
            elif between(cap, 6000, 7000):
                frame = cv2.GaussianBlur(frame,(15,15),5)
                write_text(frame, "Gaussian filter - kernel (15x15) - sigma 5", (100,50), size, color)
            elif between(cap, 7000, 8000):
                frame = cv2.GaussianBlur(frame,(15,15),10)   
                write_text(frame, "Gaussian filter - kernel (15x15) - sigma 10", (100,50), size, color)
            elif between(cap, 8000, 9000):
                frame = cv2.bilateralFilter(frame,15,5,5)
                write_text(frame, "Bilateral filter - diameter 15 - sigma spatial 5 - sigma color 5", (100,50), size, color)
                display_conclusion(frame)
            elif between(cap, 9000, 10000):
                frame = cv2.bilateralFilter(frame,15,70,70)
                write_text(frame, "Bilateral filter - diameter 15 - sigma spatial 70 - sigma color 70", (100,50), size, color)
                display_conclusion(frame)
            elif between(cap, 10000, 11000):
                frame = cv2.bilateralFilter(frame,15,150,150)
                write_text(frame, "Bilateral filter - diameter 15 - sigma spatial 150 - sigma color 150", (100,50), size, color)
                display_conclusion(frame)
            elif between(cap, 11000, 12000):
                frame = cv2.bilateralFilter(frame,15,300,300)
                write_text(frame, "Bilateral filter - diameter 15 - sigma spatial 300 - sigma color 300", (100,50), size, color)
                display_conclusion(frame)
            elif between(cap, 12000, 14500):
                #Grab in RGB
                lower = np.array([5, 30, 90])
                upper = np.array([60, 160, 260])
                
                #threshold between lower and upper
                frame_threshed = cv2.inRange(frame, lower, upper)
                frame = convert_to_displayable_frame(frame,frame_threshed)
                write_text(frame, "Grab object in RGB thanks to thresholding", (100,50), size, color)
            elif between(cap, 14500, 17000):
                #Grab in HSV
                lower = np.array([8, 50, 50],np.uint8)
                upper = np.array([17, 255, 255],np.uint8)
                
                hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                
                frame_threshed = cv2.inRange(hsv_img, lower, upper)
                frame = convert_to_displayable_frame(frame,frame_threshed)
                write_text(frame, "Grab object in HSV thanks to thresholding", (100,50), size, color)
            elif between(cap, 17000, 18500):
                lower = np.array([8, 50, 50],np.uint8)
                upper = np.array([17, 255, 255],np.uint8)
                
                hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                
                frame_threshed = cv2.inRange(hsv_img, lower, upper)
                
                kernel = np.ones((10,10),np.uint8)
                            
                #Morphological operation
                frame_open = cv2.morphologyEx(frame_threshed, cv2.MORPH_OPEN, kernel)
                
                frame = convert_to_displayable_frame(frame,frame_threshed)
                
                #Replace improvement in blue
                frame[frame_open > 0] = (255, 0, 0)
                write_text(frame, "Grab object in HSV - Improvement in blue with opening operation", (100,50), size, color)
                write_text(frame, "We notice that we are more focused on the object (less background detection)", (100,80), size, color)
            elif between(cap, 18500, 20000):
                lower = np.array([8, 50, 50],np.uint8)
                upper = np.array([17, 255, 255],np.uint8)
                
                hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                
                frame_threshed = cv2.inRange(hsv_img, lower, upper)
                
                kernel = np.ones((10,10),np.uint8)
                
                #Morphological operations
                frame_erode = cv2.erode(frame_threshed,kernel,iterations = 2)
                frame_erode_dilate = cv2.dilate(frame_erode,kernel,iterations = 4)
                frame = convert_to_displayable_frame(frame,frame_threshed)

                #Replace improvement in blue
                frame[frame_erode_dilate > 0] = (255, 0, 0)
                write_text(frame, "Grab object in HSV - Improvement in blue with erosion (2 iterations) + dilation (4 iterations)", (100,50), size, color)
                write_text(frame, "We notice that we are completely focused on the object (no background detection)", (100,80), size, color)
            elif between(cap, 20000,21000):
                #Gray image
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
                #Blur for better detection
                frame_blur = cv2.GaussianBlur(frame_gray, (3,3), 0)
                
                #X detection
                sobel_x = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
                
                hsv = create_hsv_color_edge(sobel_x, np.zeros((810,1440)))

                #Convert hsv to  bgr
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) 
                write_text(frame, "Vertical edges (kernel 3x3)", (100,50), size, color)
            elif between(cap, 21000,22000):
                #Gray image
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
                #Blur for better detection
                frame_blur = cv2.GaussianBlur(frame_gray, (3,3), 0)
                
                #Y detection
                sobel_y = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
                
                hsv = create_hsv_color_edge(sobel_y, np.zeros((810,1440)))

                #Convert hsv to  bgr
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                write_text(frame, "Horizontal edges (kernel 3x3)", (100,50), size, color)
            elif between(cap, 22000,23000):
                #Gray image
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
                #Blur for better detection
                frame_blur = cv2.GaussianBlur(frame_gray, (3,3), 0)
                
                #X and Y detection
                sobel_x = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
                sobel_y = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
                
                hsv = create_hsv_color_edge(sobel_x, sobel_y)
                
                #Convert hsv to  bgr
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                write_text(frame,  "Combination of Horizontal and Vertical edges (kernel 3x3)", (100,50), size, color)
            elif between(cap, 23000,24000):
                #Gray image
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
                #Blur for better detection
                frame_blur = cv2.GaussianBlur(frame_gray, (3,3), 0)
                
                #X and Y detection
                sobel_x = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
                sobel_y = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
                
                hsv = create_hsv_color_edge(sobel_x, sobel_y)

                #Convert hsv to  bgr
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                write_text(frame,  "Combination of Horizontal and Vertical edges (kernel 5x5)", (100,50), size, color)
            elif between(cap, 24000,25000):
                #Gray image
                frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                
                #Blur for better detection
                frame_blur = cv2.GaussianBlur(frame_gray, (3,3), 0)
                
                #X and Y detection
                sobel_x = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=7)
                sobel_y = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=7)
                
                hsv = create_hsv_color_edge(sobel_x, sobel_y)

                #Convert hsv to  bgr
                frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                write_text(frame,  "Combination of Horizontal and Vertical edges (kernel 7x7)", (100,50), size, color)
            elif between(cap, 25000,26000):
                hough_transform_circle(frame, 2, 1/16, 40, 24, 30, 36)
                write_text(frame,  "Inverse ratio of the accumulator resolution to the image resolution: dp = 2", (100,50), size, color)
            elif between(cap, 26000,27000):
                hough_transform_circle(frame, 1, 1/16, 40, 24, 30, 36)
                write_text(frame,  "Inverse ratio of the accumulator resolution to the image resolution: dp = 1 (better results with this setting)", (100,50), size, color)
            elif between(cap, 27000,28000):
                hough_transform_circle(frame, 1, 1/100, 40, 24, 30, 36)
                write_text(frame,  "Minimum distance between the centers of the detected circles: minDist proportional to 1/100", (100,50), size, color)
            elif between(cap, 28000,29000):
                hough_transform_circle(frame, 1, 1/16, 100, 20, 36, 42)
                write_text(frame,  "Minimum distance between the centers of the detected circles: minDist proportional to 1/16", (100,50), size, color)
                write_text(frame,  "(better results with this setting)", (100,70), size, color)
            elif between(cap, 29000,30000):
                hough_transform_circle(frame, 1, 1/16, 5, 24, 30, 36)
                write_text(frame,  "Higher threshold of the Canny edge detector: param1 = 5", (100,50), size, color)
            elif between(cap, 30000,31000):
                hough_transform_circle(frame, 1, 1/16, 40, 24, 30, 36)
                write_text(frame,  "Higher threshold of the Canny edge detector: param1 = 40 (better results with this setting)", (100,50), size, color)
            elif between(cap, 31000,32000):
                hough_transform_circle(frame, 1, 1/16, 40, 4, 30, 36)
                write_text(frame, "Accumulator threshold for the circle centers at the detection stage (the smaller, the worse): param2 = 4", (100,50), size, color)
            elif between(cap, 32000,33000):
                hough_transform_circle(frame, 1, 1/16, 40, 24, 30, 36)
                write_text(frame, "Accumulator threshold for the circle centers at the detection stage (the smaller, the worse): param2 = 24", (100,50), size, color)
                write_text(frame, "(better results with this setting)", (100,70), size, color)
            elif between(cap, 33000,34000):
                hough_transform_circle(frame, 1, 1/16, 40, 24, 0, 100)
                write_text(frame, "Minimum circle radius: minRadius = 0 | Maximum circle radius: maxRadius = 100", (100,50), size, color)
            elif between(cap, 34000,35000):
                hough_transform_circle(frame, 1, 1/16, 40, 24, 30, 36)
                write_text(frame, "Minimum circle radius: minRadius = 30 | Maximum circle radius: maxRadius = 36", (100,50), size, color)
                write_text(frame, "(better results with this setting)", (100,70), size, color)

                write_text(frame, "Good parameters: dp = 1 | minDist proportional to 1/16 |", (100,100), size, color)
                write_text(frame, "param1 = 40 | param2 = 24 | minRadius = 30 | maxRadius = 36", (100,120), size, color)
            elif between(cap, 35000,37000):
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                template = cv2.imread('tool.jpg',0)    
                
                w, h = template.shape[::-1]

                #Apply template Matching
                method = cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(img,template,method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                #Draw rectangle
                cv2.rectangle(frame,top_left, bottom_right, 255, 2)
                write_text(frame, "Find object: tool - box", (100,50), size, color)
            elif between(cap, 37000,40000):
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                template = cv2.imread('tool.jpg',0)    
                
                w, h = template.shape[::-1]

                #Apply template Matching
                method = cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(img,template,method)
 
                #Resize to good dimensions
                res = cv2.resize(res, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_LINEAR)
                
                #Set good scale
                res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                
                frame = convert_to_displayable_frame(frame, res)
                write_text(frame, "Find object: tool - grayscale map", (100,50), size, color)
            elif between(cap,40000,42000):
                #Update tracker
                ret_track, bbox = tracker.update(frame)
                
                #Draw box
                if ret:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                else:
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                write_text(frame, "Tracking of an object (we follow the orange)", (100,50), size, color)
            elif between(cap,42000,44500):
                #Update tracker
                ret_track, bbox = tracker.update(frame)
                                
                #Draw circle
                if ret:
                    center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
                    radius = int((bbox[2]/2+1)*0.95)
                    cv2.circle(frame, center, radius, (190,198,202), -1) #Fill circles with same colour as background to make them invisible
                else:
                    cv2.circle(frame, center, radius, (190,198,202), -1)
                write_text(frame, "Make the object invisible", (100,50), size, color)
            elif between(cap,44500,48000):
                #Update tracker
                ret_track, bbox = tracker.update(frame)
                
                mask = cv2.imread("smiley.jpeg")
                mask = cv2.resize(mask, (int(bbox[2]*size_smiley), int(bbox[3]*size_smiley)), interpolation = cv2.INTER_LINEAR)
                
                try:
                    frame[int(bbox[1]):(int(bbox[1])+mask.shape[0]), int(bbox[0]):(int(bbox[0])+mask.shape[1])] = mask
                except:
                    print("Fail to replace")
                write_text(frame, "Replace digitally with another object", (100,50), size, color)
                write_text(frame, "Note: We can see that when the target leaves the screen, we cannot cover it", (100,80), size, color)
            elif between(cap, 48000,51000):
                #Update tracker
                ret_track, bbox = tracker.update(frame)
                
                mask = cv2.imread("smiley.jpeg")
                mask = cv2.resize(mask, (int(bbox[2]*size_smiley), int(bbox[3]*size_smiley)), interpolation = cv2.INTER_LINEAR)
                
                try:
                    frame[int(bbox[1]):(int(bbox[1])+mask.shape[0]), int(bbox[0]):(int(bbox[0])+mask.shape[1])] = mask
                except:
                    print("Fail to replace")
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                template = cv2.imread('eyes.jpg',0)    
                
                w, h = template.shape[::-1]            

                #Apply template Matching
                method = cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(img,template,method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                top_left = max_loc
               
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame,top_left, bottom_right, 255, 2)
                write_text(frame, "Detect eyes", (100,50), size, color)
            elif between(cap, 51000,54000):
                #Update tracker
                ret_track, bbox = tracker.update(frame)
                
                mask = cv2.imread("smiley.jpeg")
                mask = cv2.resize(mask, (int(bbox[2]*size_smiley), int(bbox[3]*size_smiley)), interpolation = cv2.INTER_LINEAR)
                
                try:
                    frame[int(bbox[1]):(int(bbox[1])+mask.shape[0]), int(bbox[0]):(int(bbox[0])+mask.shape[1])] = mask
                except:
                    print("Fail to replace")
                
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                template = cv2.imread('eyes.jpg',0)    
                
                w, h = template.shape[::-1]            

                #Apply template Matching
                method = cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(img,template,method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                top_left = max_loc
                    
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                mask1 = cv2.imread("eyes_cartoon.jpeg")
                mask1 = cv2.resize(mask1, (w, h), interpolation = cv2.INTER_LINEAR)

                try:
                    frame[int(top_left[1]):(int(top_left[1])+mask1.shape[0]), int(top_left[0]):(int(top_left[0])+mask1.shape[1])] = mask1
                except:
                    print("Fail to replace")
                write_text(frame, "Replace eyes by another object", (100,50), size, color)
            elif between(cap,54000,54100):
                #New target
                tracker.clear()
                tracker = cv2.TrackerCSRT_create()
                bbox = (826, 522, 85, 71)
                ret_track = tracker.init(frame, bbox)
            elif between(cap,54100,57000):
                #Update tracker
                ret_track, bbox = tracker.update(frame)
                
                #Draw box
                if ret:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                else:
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                write_text(frame, "We change of target (we focus on another orange)", (100,50), size, color)
                write_text(frame, "Algorithm robust against change of scale", (100,70), size, color)
            elif between(cap,57000,60000):
                #Update tracker
                ret_track, bbox = tracker.update(frame)
                
                #Draw box
                if ret:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                else:
                    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                write_text(frame, "Algorithm robust against change of rotation", (100,50), size, color)
 
            #Write time
            write_text(frame, 'Time: ' + '%.2f' % float(cap.get(cv2.CAP_PROP_POS_MSEC)/1000) + ' s', (100,20), 0.5, color)

            #Display the resulting frame
            cv2.imshow('Frame', frame)
            
            #Write frame for the output
            out.write(np.uint8(frame))

            #Press q to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        #Break 
        else:
            break

    #Release the video capture and writing object
    cap.release()
    out.release()
    
    #Close the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)