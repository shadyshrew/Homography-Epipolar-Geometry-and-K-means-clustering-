def task3_GMM():    
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt
    import numpy as np
    
    X = [[5.9, 3.2],[4.6, 2.9],[6.2,2.8],[4.7,3.2],[5.5,4.2],[5.0,3.0 ],[4.9,3.1],[6.7,3.1],[5.1,3.8],[6.0,3.0]]
    X = np.asarray(X)
    MU = [[6.2,3.2],[6.6,3.7],[6.5,3.0]]
    COV = [[[0.5, 0], [0, 0.5]],[[0.5, 0], [0, 0.5]],[[0.5, 0], [0, 0.5]]]
    #rv = multivariate_normal(MU[0],COV)
    #a = rv.pdf(X)
    
    def iterations(mu,cov,i):
        cva = classify_and_plot(mu,cov)
        #plt.savefig('task3_gmm_iter'+str(i+1)+'_a.jpg')
        #plt.clf()
        mu = recompute(cva,mu)
        #plot_mu(mu[0],mu[1],mu[2])
        #plt.savefig('task3_gmm_iter'+str(i+1)+'_b.jpg')
        #plt.clf()
        print('The recomputed MU are:')
        for i in mu:
            print(i)
        return mu,cov
    
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
    
    def classify_and_plot(mu,cov):
        #Initializing Centres
        #plot_mu(mu[0],mu[1],mu[2])    
        #for i in X:
        #    plt.text(i[0]+0.03,i[1], s = '('+str(i[0])+','+str(i[1])+')',fontsize = 8)
        class_vec = []
    
        pdfs = []
        for i in range(len(mu)):
            pdfs.append([])
        for i in range(len(mu)):
            pdfs[i].append(multivariate_normal(mu[i],cov[i]).pdf(X))
        print(pdfs)
        #print(max(pdfs[0][0][8]))
        for i in range(len(X)):
            if(max(pdfs[0][0][i],pdfs[1][0][i],pdfs[2][0][i]) == pdfs[0][0][i]):
                class_vec.append(mu[0])
            elif(max(pdfs[0][0][i],pdfs[1][0][i],pdfs[2][0][i]) == pdfs[1][0][i]):
                class_vec.append(mu[1])
            elif(max(pdfs[0][0][i],pdfs[1][0][i],pdfs[2][0][i]) == pdfs[2][0][i]):
                class_vec.append(mu[2])
        '''
        for i,j in zip(class_vec,X):
            if(i == mu[0]):
                plt.scatter(j[0],j[1],s = 50,c = 'w',marker = '^', linewidths = 1,edgecolor = 'r' )
            if(i == mu[1]):
                plt.scatter(j[0],j[1],s = 50,c = 'w',marker = '^', linewidths = 1,edgecolor = 'g' )
            if(i == mu[2]):
                plt.scatter(j[0],j[1],s = 50,c = 'w',marker = '^', linewidths = 1,edgecolor = 'b' )
        '''
        print('The classification vector is: ')
        for i in class_vec:
            print(i)
        return class_vec
    
    for i in range(1):
        MU,COV = iterations(MU,COV,i)

#call the main function
task3_GMM()
