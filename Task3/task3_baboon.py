#Shreyas N
#UBIT Name: sn58
#Person No. - 50289736
def task3_baboon():
    import cv2
    import numpy as np
    import math
    from matplotlib import pyplot as plt
    import time
    
    UBID = '50289736'
    np.random.seed(sum([ord(c) for c in UBID]))
    print('Quantizing Colors')
    def display(name,img):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    s = time.time()
    baboon = cv2.imread('baboon.jpg')
    h,w,c = baboon.shape
    MU = []
    baboon = baboon.reshape((baboon.shape[0]*baboon.shape[1]),3)
    temp = baboon.copy()
    np.random.shuffle(temp)
    
    def draw_baboon(clusters):
        
        MU = []
        for i,j in zip(range(clusters),temp):
            MU.append(temp[i])
        MU = np.asarray(MU)
        
        X = baboon
        def iterations(mu):
            cva = []
            cva,kc = classify_and_plot(mu)
            mu = recompute(cva,mu,kc)
            return mu,cva
        def recompute(class_vector,mu,kc):
        
            nmu = []
            for i in range(len(mu)):
                clus = np.asarray(kc[i])
                meaney = np.mean(clus, axis = 0)
                nmu.append(meaney.tolist())
            return np.asarray(nmu)    
            
        def classify_and_plot(mu):  
            class_vec = []
            kc = []
            for i in range(clusters):
                kc.append([])
            for x in X:
                euc = []
                for i in range(len(mu)):
                    euc.append(np.sqrt(sum(np.square(np.subtract(mu[i],x)))))
                t = euc.index(min(euc))
                class_vec.append(mu[t])
                #print(t)
                kc[t].append(x)
            return np.asarray(class_vec),kc
    
        for i in range(20):
            MU,image = iterations(np.asarray(MU))
        image = image.reshape(h,w,3)
        cv2.imwrite('task3_baboon_'+str(clusters)+'.jpg',image)
        print('Time taken for k = '+ str(clusters))
        taime = time.time() - s
        print(taime)
    a = [3,5,10,20]
    #a = [3]
    for i in a:
        draw_baboon(i)

#call the main function
task3_baboon()