import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.linalg import svd
from statistics import mean 

filename = '../../SAheart.csv'
df = pd.read_csv(filename)

raw_data = df.get_values()
raw_data=np.concatenate((raw_data[:,1:4],raw_data[:,5:11]),axis=1)

famHistIndex=3
classLabels = raw_data[:,famHistIndex]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
raw_data[:,famHistIndex]=np.array([classDict[cl] for cl in classLabels])
attributeNames = list((np.asarray(df.columns[1:4]))) +list(np.asarray(df.columns[5:11]))

# Data attributes to be plotted
i = 0
j = 1
f = figure()
title('data')
for c in range(2):
    # select indices belonging to class c:
    class_mask = raw_data[:,-1]==c
    plot(raw_data[class_mask,i], raw_data[class_mask,j], 'o',alpha=.3)

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])
show()

N, M = raw_data.shape
for c in range(M):
    n, bins, patches = plt.hist(x=raw_data[:,c], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    title(attributeNames[c])
    show()

X2 = np.empty((N, M-1))
for i, col_id in enumerate(range(0, M-1)):
    X2[:, i] = np.asarray(list(raw_data[:,i]))

Y = X2 - np.ones((N,1))*X2.mean(axis=0)
Y = Y*(1/np.std(Y,0))

covarianceMatrix=np.corrcoef(Y,rowvar=False)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)
# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T    
# Project the centered data onto principal component space
Z = Y @ V
# Indices of the principal components to be plotted
i = 0
j = 1
# Plot PCA of the data
f = figure()
title('PCA')
#Z = array(Z)
for c in range(2):
    class_mask = raw_data[:,-1]==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

show()
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA Component Coefficients')
plt.show()

print('PC2:')
print(V[:,1].T)