import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



def detector(image,obj):
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_obj = obj#cv2.cvtColor(obj,cv2.COLOR_BGR2GRAY)
    
    gray_img = np.float64(gray_img)
    gray_obj = np.float64(gray_obj)
   
    
    n_components = 150
    pca = PCA(n_components=n_components , svd_solver = 'randomized').fit(gray_img)
    
    pca_img = pca.transform(gray_img)
    pca_obj = pca.transform(gray_obj)
    
   
    Kmeans_img =KMeans(n_clusters = 3,random_state =0).fit(pca_img)
    Kmeans_obj =KMeans(n_clusters = 3,random_state =0).fit(pca_obj)
    
    counter = 0
    temp = Kmeans_img.labels_
    temp2 = Kmeans_obj.labels_
    
    print temp
    for i in range(min(len(temp),len(temp2))):
        if temp[i] != temp2[i]:
            counter += 1
    return (100-counter)


cap = cv2.VideoCapture(0)

# Load our image template, this is our reference image

image_template = cv2.imread('images/box_in_scene.png', 0) 

while True:

    # Get webcam images
    ret, frame = cap.read()

    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Define ROI Box Dimensions
    top_left_x = width / 3
    top_left_y = (height / 2) + (height / 4)
    bottom_right_x = (width / 3) * 2
    bottom_right_y = (height / 2) - (height / 4)
    
    # Draw rectangular window for our region of interest   
    cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)
    
    # Crop window of observation we defined above
    
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]
    
    # Flip frame orientation horizontally
    frame = cv2.flip(frame,1)
    
    # Get number of SIFT matches
    matches = detector(cropped, image_template)

    # Display status string showing the current no. of matches 
    cv2.putText(frame,str(matches),(450,450), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),1)
    
    # Our threshold to indicate object deteciton
    # We use 10 since the SIFT detector returns little false positves
    threshold = 90
    
    # If matches exceed our threshold then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)
    
    cv2.imshow('Object Detector using SIFT', frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()   
