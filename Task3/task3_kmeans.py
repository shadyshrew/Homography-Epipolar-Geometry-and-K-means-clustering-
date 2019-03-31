#Shreyas N
#UBIT Name: sn58
#Person No. - 50289736
def task3_kmeans():
    import cv2
    import numpy as np
    import math
    from matplotlib import pyplot as plt
    UBID = '50289736'
    np.random.seed(sum([ord(c) for c in UBID]))
    
    
    
    #The datapoints matrix is 
    X = [[5.9, 3.2],[4.6, 2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0 ],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]]
    X = np.asarray(X)
    MU = [[6.2,3.2],[6.6,3.7],[6.5,3.0]]
    
    def iterations(mu,i):
        cva = classify_and_plot(mu[0],mu[1],mu[2])
        plt.savefig('task3_iter'+str(i+1)+'_a.jpg')
        plt.clf()
        mu = recompute(cva,mu)
        plot_mu(mu[0],mu[1],mu[2])
        plt.savefig('task3_iter'+str(i+1)+'_b.jpg')
        plt.clf()
        print('The recomputed MU are:')
        for i in mu:
            print(i)
        return mu
    def recompute(class_vector,mu):
        x1 = 0
        x2 = 0
        x3 = 0
        y1 = 0
        y2 = 0
        y3 = 0
        count1 = 0
        count2 = 0
        count3 = 0
        print(class_vector)
        for x,i,class_v in zip(X,range(0,len(X)),class_vector):
            if(class_v == mu[0]):
                x1 = x1 + x[0]
                y1 = y1 + x[1]
                count1 = count1+1
            elif(class_v == mu[1]):
                x2 = x2 + x[0]
                y2 = y2 + x[1]
                count2 = count2+1
            elif(class_v == mu[2]):
                x3 = x3 + x[0]
                y3 = y3 + x[1]
                count3 = count3+1
        
        if(count1!=0):
            x1 = round(x1/count1, 2)
            y1 = round(y1/count1, 2)
        if(count2!=0):
            x2 = round(x2/count2, 2)
            y2 = round(y2/count2, 2)
        if(count3!=0):
            x3 = round(x3/count3, 2)   
            y3 = round(y3/count3, 2)
        
        mu[0] = [x1,y1]
        mu[1] = [x2,y2]
        mu[2] = [x3,y3] 
        return mu
        
    def plot_mu(mu0,mu1,mu2):
        plt.text(mu0[0]+0.03,mu0[1], s = '('+str(mu0[0])+','+str(mu0[1])+')',fontsize = 8)
        plt.text(mu1[0]+0.03,mu1[1], s = '('+str(mu1[0])+','+str(mu1[1])+')',fontsize = 8)
        plt.text(mu2[0]+0.03,mu2[1], s = '('+str(mu2[0])+','+str(mu2[1])+')',fontsize = 8)
        
        plt.scatter(mu0[0],mu0[1],s = 50,c = 'r',marker = 'o', linewidths = 1,edgecolor = 'r' )
        plt.scatter(mu1[0],mu1[1],s = 50,c = 'g',marker = 'o', linewidths = 1,edgecolor = 'g' )
        plt.scatter(mu2[0],mu2[1],s = 50,c = 'b',marker = 'o', linewidths = 1,edgecolor = 'b' )
        
    def classify_and_plot(mu0,mu1,mu2):
        #Initializing Centres
        plot_mu(mu0,mu1,mu2)    
        for i in X:
            plt.text(i[0]+0.03,i[1], s = '('+str(i[0])+','+str(i[1])+')',fontsize = 8)
        class_vec = []
        for x in X:
            euc1 = np.sqrt((mu0[0]-x[0])**2 + (mu0[1]-x[1])**2)
            euc2 = np.sqrt((mu1[0]-x[0])**2 + (mu1[1]-x[1])**2)
            euc3 = np.sqrt((mu2[0]-x[0])**2 + (mu2[1]-x[1])**2)
            if(min(euc1,euc2,euc3) == euc1):
                class_vec.append(mu0)
            elif(min(euc1,euc2,euc3) == euc2):
                class_vec.append(mu1)
            elif(min(euc1,euc2,euc3) == euc3):
                class_vec.append(mu2)
        
        for i,j in zip(class_vec,X):
            if(i == mu0):
                plt.scatter(j[0],j[1],s = 50,c = 'w',marker = '^', linewidths = 1,edgecolor = 'r' )
            if(i == mu1):
                plt.scatter(j[0],j[1],s = 50,c = 'w',marker = '^', linewidths = 1,edgecolor = 'g' )
            if(i == mu2):
                plt.scatter(j[0],j[1],s = 50,c = 'w',marker = '^', linewidths = 1,edgecolor = 'b' )
        print('The classification vector is: ')
        for i in class_vec:
            print(i)
        return class_vec
    
    for i in range(2):
        MU = iterations(MU,i)
    #########babooooooon##########  
task3_kmeans()



