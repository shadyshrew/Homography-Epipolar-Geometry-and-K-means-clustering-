def task3_GMMb():  
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import numpy as np
    import math
    from matplotlib import pyplot as plt
    import time
    import csv
    from matplotlib.patches import Ellipse
    UBID = '50289736'
    np.random.seed(sum([ord(c) for c in UBID]))
    X = []
    
    t = []
    with open('old_faithful.csv','r') as f:
        reader = csv.reader(f)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            t.append(dataRow)
    X = t
    #print(X)
    MU = [[4.0,81],[2.0,57],[4.0,71]]
    COV = np.array([[[1.3,13.92],[13.98,184.82]],[[1.3,13.92],[13.98,184.82]],[[1.3,13.92],[13.98,184.82]]])
    clusters = 3
    #X = [[3.5999999046325684, 79.0], [1.7999999523162842, 54.0], [3.3329999446868896, 74.0], [2.2829999923706055, 62.0], [4.5329999923706055, 85.0], [2.882999897003174, 55.0], [4.699999809265137, 88.0], [3.5999999046325684, 85.0], [1.9500000476837158, 51.0], [4.349999904632568, 85.0], [1.8329999446868896, 54.0], [3.9170000553131104, 84.0], [4.199999809265137, 78.0], [1.75, 47.0], [4.699999809265137, 83.0], [2.1670000553131104, 52.0], [1.75, 62.0], [4.800000190734863, 84.0], [1.600000023841858, 52.0], [4.25, 79.0], [1.7999999523162842, 51.0], [1.75, 47.0], [3.450000047683716, 78.0], [3.066999912261963, 69.0], [4.5329999923706055, 74.0], [3.5999999046325684, 83.0], [1.9670000076293945, 55.0], [4.083000183105469, 76.0], [3.8499999046325684, 78.0], [4.433000087738037, 79.0], [4.300000190734863, 73.0], [4.4670000076293945, 77.0], [3.367000102996826, 66.0], [4.0329999923706055, 80.0], [3.8329999446868896, 74.0], [2.0169999599456787, 52.0], [1.8669999837875366, 48.0], [4.833000183105469, 80.0], [1.8329999446868896, 59.0], [4.7829999923706055, 90.0], [4.349999904632568, 80.0], [1.8830000162124634, 58.0], [4.566999912261963, 84.0], [1.75, 58.0], [4.5329999923706055, 73.0], [3.316999912261963, 83.0], [3.8329999446868896, 64.0], [2.0999999046325684, 53.0], [4.632999897003174, 82.0], [2.0, 59.0], [4.800000190734863, 75.0], [4.716000080108643, 90.0], [1.8329999446868896, 54.0], [4.833000183105469, 80.0], [1.7330000400543213, 54.0], [4.882999897003174, 83.0], [3.7170000076293945, 71.0], [1.6670000553131104, 64.0], [4.566999912261963, 77.0], [4.316999912261963, 81.0], [2.2330000400543213, 59.0], [4.5, 84.0], [1.75, 48.0], [4.800000190734863, 82.0], [1.8170000314712524, 60.0], [4.400000095367432, 92.0], [4.166999816894531, 78.0], [4.699999809265137, 78.0], [2.066999912261963, 65.0], [4.699999809265137, 73.0], [4.0329999923706055, 82.0], [1.9670000076293945, 56.0], [4.5, 79.0], [4.0, 71.0], [1.9830000400543213, 62.0], [5.066999912261963, 76.0], [2.0169999599456787, 60.0], [4.566999912261963, 78.0], [3.882999897003174, 76.0], [3.5999999046325684, 83.0], [4.132999897003174, 75.0], [4.333000183105469, 82.0], [4.099999904632568, 70.0], [2.632999897003174, 65.0], [4.066999912261963, 73.0], [4.933000087738037, 88.0], [3.950000047683716, 76.0], [4.517000198364258, 80.0], [2.1670000553131104, 48.0], [4.0, 86.0], [2.200000047683716, 60.0], [4.333000183105469, 90.0], [1.8669999837875366, 50.0], [4.816999912261963, 78.0], [1.8329999446868896, 63.0], [4.300000190734863, 72.0], [4.666999816894531, 84.0], [3.75, 75.0], [1.8669999837875366, 51.0], [4.900000095367432, 82.0], [2.4830000400543213, 62.0], [4.367000102996826, 88.0], [2.0999999046325684, 49.0], [4.5, 83.0], [4.050000190734863, 81.0], [1.8669999837875366, 47.0], [4.699999809265137, 84.0], [1.7829999923706055, 52.0], [4.849999904632568, 86.0], [3.683000087738037, 81.0], [4.732999801635742, 75.0], [2.299999952316284, 59.0], [4.900000095367432, 89.0], [4.416999816894531, 79.0], [1.7000000476837158, 59.0], [4.632999897003174, 81.0], [2.316999912261963, 50.0], [4.599999904632568, 85.0], [1.8170000314712524, 59.0], [4.416999816894531, 87.0], [2.617000102996826, 53.0], [4.066999912261963, 69.0], [4.25, 77.0], [1.9670000076293945, 56.0], [4.599999904632568, 88.0], [3.7669999599456787, 81.0], [1.9170000553131104, 45.0], [4.5, 82.0], [2.2669999599456787, 55.0], [4.650000095367432, 90.0], [1.8669999837875366, 45.0], [4.166999816894531, 83.0], [2.799999952316284, 56.0], [4.333000183105469, 89.0], [1.8329999446868896, 46.0], [4.382999897003174, 82.0], [1.8830000162124634, 51.0], [4.933000087738037, 86.0], [2.0329999923706055, 53.0], [3.7330000400543213, 79.0], [4.232999801635742, 81.0], [2.2330000400543213, 60.0], [4.5329999923706055, 82.0], [4.816999912261963, 77.0], [4.333000183105469, 76.0], [1.9830000400543213, 59.0], [4.632999897003174, 80.0], [2.0169999599456787, 49.0], [5.099999904632568, 96.0], [1.7999999523162842, 53.0], [5.0329999923706055, 77.0], [4.0, 77.0], [2.4000000953674316, 65.0], [4.599999904632568, 81.0], [3.566999912261963, 71.0], [4.0, 70.0], [4.5, 81.0], [4.083000183105469, 93.0], [1.7999999523162842, 53.0], [3.9670000076293945, 89.0], [2.200000047683716, 45.0], [4.150000095367432, 86.0], [2.0, 58.0], [3.8329999446868896, 78.0], [3.5, 66.0], [4.583000183105469, 76.0], [2.367000102996826, 63.0], [5.0, 88.0], [1.9329999685287476, 52.0], [4.617000102996826, 93.0], [1.9170000553131104, 49.0], [2.0829999446868896, 57.0], [4.583000183105469, 77.0], [3.3329999446868896, 68.0], [4.166999816894531, 81.0], [4.333000183105469, 81.0], [4.5, 73.0], [2.4170000553131104, 50.0], [4.0, 85.0], [4.166999816894531, 74.0], [1.8830000162124634, 55.0], [4.583000183105469, 77.0], [4.25, 83.0], [3.7669999599456787, 83.0], [2.0329999923706055, 51.0], [4.433000087738037, 78.0], [4.083000183105469, 84.0], [1.8329999446868896, 46.0], [4.416999816894531, 83.0], [2.183000087738037, 55.0], [4.800000190734863, 81.0], [1.8329999446868896, 57.0], [4.800000190734863, 76.0], [4.099999904632568, 84.0], [3.9660000801086426, 77.0], [4.232999801635742, 81.0], [3.5, 87.0], [4.366000175476074, 77.0], [2.25, 51.0], [4.666999816894531, 78.0], [2.0999999046325684, 60.0], [4.349999904632568, 82.0], [4.132999897003174, 91.0], [1.8669999837875366, 53.0], [4.599999904632568, 78.0], [1.7829999923706055, 46.0], [4.367000102996826, 77.0], [3.8499999046325684, 84.0], [1.9329999685287476, 49.0], [4.5, 83.0], [2.382999897003174, 71.0], [4.699999809265137, 80.0], [1.8669999837875366, 49.0], [3.8329999446868896, 75.0], [3.4170000553131104, 64.0], [4.232999801635742, 76.0], [2.4000000953674316, 53.0], [4.800000190734863, 94.0], [2.0, 55.0], [4.150000095367432, 76.0], [1.8669999837875366, 50.0], [4.267000198364258, 82.0], [1.75, 54.0], [4.482999801635742, 75.0], [4.0, 78.0], [4.117000102996826, 79.0], [4.083000183105469, 78.0], [4.267000198364258, 78.0], [3.9170000553131104, 70.0], [4.550000190734863, 79.0], [4.083000183105469, 70.0], [2.4170000553131104, 54.0], [4.183000087738037, 86.0], [2.2170000076293945, 50.0], [4.449999809265137, 90.0], [1.8830000162124634, 54.0], [1.850000023841858, 54.0], [4.2829999923706055, 77.0], [3.950000047683716, 79.0], [2.3329999446868896, 64.0], [4.150000095367432, 75.0], [2.3499999046325684, 47.0], [4.933000087738037, 86.0], [2.9000000953674316, 63.0], [4.583000183105469, 85.0], [3.8329999446868896, 82.0], [2.0829999446868896, 57.0], [4.367000102996826, 82.0], [2.132999897003174, 67.0], [4.349999904632568, 74.0], [2.200000047683716, 54.0], [4.449999809265137, 83.0], [3.566999912261963, 73.0], [4.5, 73.0], [4.150000095367432, 88.0], [3.816999912261963, 80.0], [3.9170000553131104, 71.0], [4.449999809265137, 83.0], [2.0, 56.0], [4.2829999923706055, 79.0], [4.767000198364258, 78.0], [4.5329999923706055, 84.0], [1.850000023841858, 58.0], [4.25, 83.0], [1.9830000400543213, 43.0], [2.25, 60.0], [4.75, 75.0], [4.117000102996826, 81.0], [2.1500000953674316, 46.0], [4.416999816894531, 90.0], [1.8170000314712524, 46.0], [4.4670000076293945, 74.0]]
    
    def plot_point_cov(points, nstd=2, ax=None, **kwargs):
        pos = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
    
    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
        
        if ax is None:
            ax = plt.gca()
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
        ax.add_artist(ellip)
        return ellip
    
    def iterations(mu,cov,i):
        cva,kc = classify_and_plot(mu,cov)
        plt.savefig('task3_gmm_iter'+str(i+1)+'_a.jpg')
        plt.clf()
        mu,cov = recompute(cva,mu,kc,cov)
        #plot_mu(mu[0],mu[1],mu[2])
        #plt.savefig('task3_gmm_iter'+str(i+1)+'_b.jpg')
        #plt.clf()
        print('The recomputed MU are:')
        for i in mu:
            print(i)
            print('\n')
        print('The recomputed Covariances are:')
        for i in cov:    
            print(i)
            print('\n')
        return mu,cov
    
    def recompute(class_vector,mu,kc,cov):
        
            nmu = []
            for i in range(len(mu)):
                clus = np.asarray(kc[i])
                meaney = np.mean(clus, axis = 0)
                nmu.append(meaney.tolist())
            nmu = np.asarray(nmu)
            for i in range(len(mu)):
                cov[i] = np.cov(kc[i],rowvar = False)
            #print(cov)
            return nmu,cov
    
    def plot_mu(mu0,mu1,mu2):
        plt.text(mu0[0]+0.03,mu0[1], s = '('+str(mu0[0])+','+str(mu0[1])+')',fontsize = 8)
        plt.text(mu1[0]+0.03,mu1[1], s = '('+str(mu1[0])+','+str(mu1[1])+')',fontsize = 8)
        plt.text(mu2[0]+0.03,mu2[1], s = '('+str(mu2[0])+','+str(mu2[1])+')',fontsize = 8)
        
        plt.scatter(mu0[0],mu0[1],s = 50,c = 'r',marker = 'o', linewidths = 1,edgecolor = 'r' )
        plt.scatter(mu1[0],mu1[1],s = 50,c = 'g',marker = 'o', linewidths = 1,edgecolor = 'g' )
        plt.scatter(mu2[0],mu2[1],s = 50,c = 'b',marker = 'o', linewidths = 1,edgecolor = 'b' )
    
    def classify_and_plot(mu,cov):
        class_vec = []
    
        pdfs = []
        for i in range(len(mu)):
            pdfs.append([])
        for i in range(len(mu)):
            pdfs[i].append(multivariate_normal.pdf(X, mu[i], cov[i]))
        kc = []
        for i in range(clusters):
            kc.append([])
            
        for i in range(len(X)):
            mac = max(pdfs[0][0][i],pdfs[1][0][i],pdfs[2][0][i])
            if( mac == pdfs[0][0][i]):
                class_vec.append(mu[0])
                kc[0].append(X[i])
            elif(mac == pdfs[1][0][i]):
                class_vec.append(mu[1])
                kc[1].append(X[i])
            elif(mac == pdfs[2][0][i]):
                class_vec.append(mu[2])
                kc[2].append(X[i])
    
        kc1 = np.asarray(kc[0])
        kc2 = np.asarray(kc[1])
        kc3 = np.asarray(kc[2])
    
        plt.scatter(kc1[:,0].ravel(),kc1[:,1].ravel(),s = 30,c = 'w',marker = '^', linewidths = 1,edgecolor = 'r' )
        plt.scatter(kc2[:,0].ravel(),kc2[:,1].ravel(),s = 30,c = 'w',marker = '^', linewidths = 1,edgecolor = 'g' )
        plt.scatter(kc3[:,0].ravel(),kc3[:,1].ravel(),s = 30,c = 'w',marker = '^', linewidths = 1,edgecolor = 'b' )
        
        #cov[0] = np.cov(kc1,rowvar = False)
        pos1 = kc1.mean(axis = 0)
        
        #cov[1] = np.cov(kc2,rowvar = False)
        pos2 = kc2.mean(axis = 0)
        
        #cov[2] = np.cov(kc3,rowvar = False)
        pos3 = kc3.mean(axis = 0)
        
        
        plot_cov_ellipse(cov[0],pos1,nstd = 3, alpha = 0.5, color = 'red')
        plot_cov_ellipse(cov[1],pos2,nstd = 3, alpha = 0.5, color = 'green')
        plot_cov_ellipse(cov[2],pos3,nstd = 3, alpha = 0.5, color = 'blue')
        
        return class_vec,kc
    
    for i in range(5):
        MU,COV = iterations(MU,COV,i)
#call the function
task3_GMMb()