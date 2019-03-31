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
def task2():
    import cv2
    import numpy as np
    import math
    UBID = '50289736'
    np.random.seed(sum([ord(c) for c in UBID]))
    
    def draw_epi(p1,p2,line,i1,i2,ic1,ic2,colo):
        h = i1.shape[0]
        w = i1.shape[1]
        for c,pp1,pp2,l in zip(colo,p1,p2,line):
            #print(l)
            x0,y0 = map(int, [0, -l[2]/l[1]])
            x1,y1 = map(int, [w, -(l[2]+l[0]*w)/l[1]])
            ic1 = cv2.line(ic1, (x0,y0), (x1,y1), (np.int(c[0]),np.int(c[1]),np.int(c[2])),1)
            ic1 = cv2.circle(ic1,tuple(pp1),3,(np.int(c[0]),np.int(c[1]),np.int(c[2])),-1)
        return ic1
        
    def display(name,img):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    imc1 = cv2.imread('tsucuba_left.png')
    imc2 = cv2.imread('tsucuba_right.png')
    img1 = cv2.imread('tsucuba_left.png',0)
    img2 = cv2.imread('tsucuba_right.png',0)
    
    #Compute the panaramic image after warping the first image
    height1,width1 = img1.shape
    height2,width2 = img2.shape
    # Creating a SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()
    #Gettig keypoints
    k1,d1 = sift.detectAndCompute(img1, None)
    k2,d2 = sift.detectAndCompute(img2, None)
    #Overlay keypoints
    i1 = imc1
    i2 = imc2
    imgk21 = cv2.drawKeypoints(i1, k1, None)
    imgk22 = cv2.drawKeypoints(i2, k2, None)
    #Save image to a file
    imgmatcher = cv2.BFMatcher()
    matches = imgmatcher.knnMatch(d1,d2,k=2)
    g = []
    for m,n in matches:
        if m.distance<0.75*n.distance:
            g.append(m)
    nomask = cv2.drawMatches(imc1,k1,imc2,k2,g,None,flags = 2)
    
    source = np.int32([ k1[m.queryIdx].pt for m in g ]).reshape(-1,1,2)
    destination = np.int32([ k2[m.trainIdx].pt for m in g ]).reshape(-1,1,2)
    
    F, mask = cv2.findFundamentalMat(source,destination,cv2.RANSAC,50.0)
    #print(F)
    matchesMask = mask.ravel().tolist()
    
    indices = []
    #imgmatch = cv2.drawMatches(imc1,k1,imc2,k2,g,None,flags = 2, matchesMask = matchesMask,matchColor = (255,0,0))
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
    #print(indices)
    #computing epilines
    color = []
    #color = tuple(np.random.randint(0,255,3).tolist())
    #for i in range(10):
        #color.append(np.random.randint(0,200,3))
    color.append([255,0,0])
    color.append([0,255,0]) 
    color.append([0,0,255])
    color.append([255,0,255])
    color.append([0,255,255])
    color.append([255,255,0])
    color.append([4,120,255])
    color.append([150,25,60])
    color.append([200,200,255])
    color.append([255,255,255])
    color.append([0,0,255])
    #color = np.int32(color)
    g1 = []
    for i in rand:
        g1.append(g[i])
    
    source = np.int32([ k1[m.queryIdx].pt for m in g1]).reshape(-1,1,2)
    destination = np.int32([ k2[m.trainIdx].pt for m in g1]).reshape(-1,1,2)
    #print(source1)
    #print(destination1)
    l2 = cv2.computeCorrespondEpilines(source.reshape(-1,1,2),1,F).reshape(-1,3)
    l1 = cv2.computeCorrespondEpilines(destination.reshape(-1,1,2),2,F).reshape(-1,3)
    '''
    for c,i,p1,p2 in zip(color,l1,source.reshape(-1,2),destination.reshape(-1,2)):
        x,y = map(int, [0,-i[2]/i[1]])
        x1,y1 = map(int,[width1, -(i[2]+i[0]*width1)/i[1]])
        img_1 = cv2.line(imc1, (x,y), (x1,y1),(np.int(c[0]),np.int(c[1]),np.int(c[2])),1)
        img_1 = cv2.circle(imc1,tuple(p1),2,(np.int(c[0]),np.int(c[1]),np.int(c[2])),-1)
        #img_2 = cv2.circle(imc2,tuple(p2),2,(np.int(c[0]),np.int(c[1]),np.int(c[2])),-1)
        
    
    
    for c,i,p1,p2 in zip(color,l1,destination.reshape(-1,2),source.reshape(-1,2)):
        x,y = map(int, [0,-i[2]/i[1]])
        x1,y1 = map(int,[width2, -(i[2]+i[0]*width2)/i[1]])
        img_3 = cv2.line(imc2, (x,y), (x1,y1),(np.int(c[0]),np.int(c[1]),np.int(c[2])),1)
        img_3 = cv2.circle(imc2,tuple(p1),2,(np.int(c[0]),np.int(c[1]),np.int(c[2])),-1)
        img_4 = cv2.circle(imc1,tuple(p2),2,(np.int(c[0]),np.int(c[1]),np.int(c[2])),-1)
    '''
    img_1 = draw_epi(source.reshape(-1,2),destination.reshape(-1,2),l1,img1,img2,imc1,imc2,color)
    img_2 = draw_epi(destination.reshape(-1,2),source.reshape(-1,2),l2,img2,img1,imc2,imc1,color)
    #mix = cv2.StereoBM_create(numDisparities=96, blockSize = 5)
    mix =  cv2.StereoBM_create(numDisparities = 64,blockSize = 15)
    #disparity = mix.compute(img1,img2)
    disparity = mix.compute(img1, img2).astype(np.float32) / 3.0
    #disparity = (disparity-32)/8
    disparity = disparity[:,60:]
    print('The Fundamental Matrix F is: \n'+str(F))
    cv2.imwrite('task2_sift1.jpg',imgk21)
    cv2.imwrite('task2_sift2.jpg',imgk22)
    cv2.imwrite('task2_matches_knn.jpg',nomask)
    #cv2.imwrite('task2_matches_knn.jpg',imgmatch)
    cv2.imwrite('task2_epi_left.jpg',img_1)
    cv2.imwrite('task2_epi_right.jpg',img_2)
    cv2.imwrite('task2_disparity.jpg',disparity)
    #print('Press enter to close window or end script')
    #input()

task2()
