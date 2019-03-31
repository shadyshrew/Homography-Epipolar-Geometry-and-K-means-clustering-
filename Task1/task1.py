#Shreyas N
#UBIT Name: sn58
#Person No. - 50289736
'''
Code referenced from
http://stamfordresearch.com/basic-sift-in-python/

https://www.programcreek.com/python/example/89309/cv2.drawKeypoints

https://stackoverflow.com/questions/46607647/sift-feature-matching-point-coordinates

https://stackoverflow.com/questions/48063525/error-with-matches1to2-with-opencv-sift

https://www.youtube.com/watch?v=MlaIWymLCD8

https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

https://stackoverflow.com/questions/33695580/selecting-random-elements-in-a-list-conditional-on-attribute

https://www.learnopencv.com/homography-examples-using-opencv-python-c/

https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective/20355545#20355545

https://docs.opencv.org/3.2.0/da/de9/tutorial_py_epipolar_geometry.html

https://programtalk.com/python-examples/cv2.computeCorrespondEpilines/

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_depthmap/py_depthmap.html

https://rdmilligan.wordpress.com/2016/05/23/disparity-of-stereo-images-with-python-and-opencv/

https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html

https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/

https://github.com/joferkington/oost paper code/blob/master/error_ellipse.py

https://www.stat.cmu.edu/˜larry/all-of-statistics/=data/faithful.dat

https://www.youtube.com/watch?v=0NMC2NfJGqo - Jordan Boyd Graber, GMM
'''
def task1():
    import cv2
    import numpy as np
    import math
    UBID = '50289736'
    np.random.seed(sum([ord(c) for c in UBID]))
    def display(name,img):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    imc1 = cv2.imread('mountain1.jpg')
    imc2 = cv2.imread('mountain2.jpg')
    img1 = cv2.imread('mountain1.jpg',0)
    img2 = cv2.imread('mountain2.jpg',0)
    #display('',img1)
    #display('',img2)
    
    # Creating a SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()
    #Gettig keypoints
    k1,d1 = sift.detectAndCompute(img1, None)
    k2,d2 = sift.detectAndCompute(img2, None)
    #Overlay keypoints
    i1 = imc1
    i2 = imc2
    imgk1 = cv2.drawKeypoints(i1, k1, None)
    imgk2 = cv2.drawKeypoints(i2, k2, None)
    #Save image to a file
    imgmatcher = cv2.BFMatcher()
    matches = imgmatcher.knnMatch(d1,d2,k=2)
    g = []
    for m,n in matches:
        if m.distance<0.75*n.distance:
            g.append(m)      
    nomask = cv2.drawMatches(imc1,k1,imc2,k2,g,None,flags = 2)
    
    #Creating a 3x3 mask for finding inliers using RANSAC
    source = np.float32([ k1[m.queryIdx].pt for m in g ]).reshape(-1,1,2)
    destination = np.float32([ k2[m.trainIdx].pt for m in g ]).reshape(-1,1,2)
    homo, mask = cv2.findHomography(source, destination, cv2.RANSAC,5.0)
    #print(homo)
    #Turn it into a form we can pass as a parameter by ravelling
    matchesMask = mask.ravel().tolist()
    
    #Compute the panaramic image after warping the first image
    height1,width1 = img1.shape
    height2,width2 = img2.shape
    
    pointssrc = np.float32([ [0,0],[0,height1],[width1,height1],[width1,0] ]).reshape(-1,1,2)
    pointsdest = np.float32([ [0,0],[0,height2],[width2,height2],[width2,0] ]).reshape(-1,1,2)
    pointssrc1 = cv2.perspectiveTransform(pointssrc, homo)
    
    points = np.concatenate((pointssrc1, pointsdest), axis=0)
    [xmin, ymin] = np.int32(points.min(axis=0).ravel() - 0.6)
    [xmax, ymax] = np.int32(points.max(axis=0).ravel() + 0.6)
    
    t = [-xmin,-ymin]
    translate = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    
    res = cv2.warpPerspective(imc1, translate.dot(homo), (xmax-xmin, ymax-ymin))
    res[t[1]:height2+t[1],t[0]:width2+t[0]] = imc2
    
    
    
    
    indices = []
    for i in range(0,len(matchesMask)):
        if(matchesMask[i] == 1):
            indices.append(i)
    np.random.shuffle(indices)
    rand = indices[:10]
    
    for i in range(0,len(matchesMask)):
        matchesMask[i] = 0
    
    for i in rand:
        matchesMask[i] = 1
    imgmatch = cv2.drawMatches(imc1,k1,imc2,k2,g,None,flags = 2, matchesMask = matchesMask,matchColor = (255,0,0))
    
    #Warping the images
    #print(img2.shape)
    imc2 = cv2.warpPerspective(imc1, homo, (img1.shape[1],img1.shape[0]))
    '''
    display('',imgk1)
    display('',imgk2)
    display('',imgmatch)
    display('',nomask)
    display('',res)
    '''
    print('The homography image H is:\n '+str(homo))
    cv2.imwrite('task1_sift1.jpg',imgk1)
    cv2.imwrite('task1_sift2.jpg',imgk2)
    cv2.imwrite('task1_matches.jpg',imgmatch)
    cv2.imwrite('task1_matches_knn.jpg',nomask)
    cv2.imwrite('task1_pano.jpg',res)
    #print('Press enter to close window or end script')
    #input()
#Call the main function
task1()