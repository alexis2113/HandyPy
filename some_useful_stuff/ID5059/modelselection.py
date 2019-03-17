#!/usr/bin/env python
# coding: utf-8



# In[3]:


import numpy as np
import scipy
from scipy.linalg import lstsq
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import pylab 
import scipy.stats as ss
import math
import numpy.matlib


# In[6]:


###
auto=pd.read_csv("auto-data.csv")

auto=auto.dropna(axis='rows')
auto.head()


# In[7]:


mpg=auto.mpg
dis=auto.displacement
horse=auto.horsepower
wei=auto.weight
ace=auto.acceleration


# In[52]:



def rk(x,z):
    return ((z-0.5)**2 - 1/12) * ((x-0.5)**2 - 1/12)/4 - ((abs(x-z)-0.5)**4 - (abs(x-z)-0.5)**2 / 2+7/240) / 24 
def rk2(x):
        return ((x-0.5)**2 - 1/12) * ((x-0.5)**2 - 1/12)/4 - ((-0.5)**4 - (-0.5)**2 / 2+7/240) / 24 

#Set up the penalized regression spline penalty matrix
def splS(xk):
    q=len(xk)+2
    S=np.matlib.zeros((q,q))
    xkm=np.fromfunction(lambda i, j: rk(xk[i],xk[j]), (q-2, q-2), dtype=int)
    S[2:,2:]=xkm
    return(S)

#Set up model matrix for cubic penalized regression spline
def splX(x,xk):
    q=len(xk)+2
    n=len(x)
    X=np.matlib.ones((n,q))
    x=x.reshape((n,1))
    X[:,1]=x
    for i in range(q-2):
        X[:,i+2]=np.apply_along_axis(rk,1,x,xk[i])
    return(X)

#Function for calculation the square root of a matrix
def matsqrt(S):
    val,vec=np.linalg.eig(S)
    rS=vec.dot(np.diag(np.sqrt(val))).dot(vec.T)
    return(rS)




def prsfit(x,y,lanbda=1,knots=3,knotsdef=0):
    x=x-min(x)
    x=x/max(x)
  #Sort x values in ascending order
    x=np.array(x)
    y=np.array(y)
    index=np.argsort(x,axis=0,kind="stable")
    y = y[index]
    x =np.sort(x,kind="stable")
    n = len(x) 
  
 
  #Calculate knot postions
    if(type(knotsdef)!=int):
        knotpos = knotsdef
    else:
        knotpos =  np.array(range(1,knots+1)) / (knots+1) 
  
  #Create Design Matrix
    q=len(knotpos)+2
    n=len(x)
    DM =np.concatenate((splX(x,knotpos), matsqrt(splS(knotpos))*math.sqrt(lanbda)), axis=0)
    y2=np.append(y,np.zeros(q))
    
 ###Calculate hat matrix using persudo-inverse of design matrix 
    hat=DM.dot(np.linalg.pinv(DM)).T
    trace=np.diag(hat)[0:n].sum()
 
   #Perform Linear Regreesion
    coef, rss, rnk, s = lstsq(DM, y2)
  #Coefficient of determination: R^2
    tss = np.sum((y-np.mean(y))**2)
  #Adjusted Coefficient of determination: R^2
    yhat = DM.dot(coef)  
    ess= tss-rss
    R2=1-(rss/tss)
    R2adj = 1 - ( (n-1)/(n-q) ) * (1-R2)
  #AIC approximation for normally distributed error
    LL=(n)*np.log(2*math.pi)+(n)*np.log(rss/n)+n
    aic=2*q+LL
    d = {'RSS':rss,
       'TSS':tss,
        "kp":knotpos,
       'R2':R2,
        'R2adj':R2adj,
        'aic':aic,
         "k":knots,
         'trace':trace,
        'yhat':yhat,
        'coef':coef,
        'ess':ess}
    return(d)


# In[41]:


from concurrent.futures import ThreadPoolExecutor
from functools import partial

## Using horsepower as example:

def Lowess(regfun,myx,myy,itimes=5):
    lad=list(gcv["lambda"][gcv.vars=="horsepower"])
    knot=list(gcv.k)
    ks=[]
    box=[]
    box2=[]
    box3=[]
    func=partial(regfun,myx,myy)
    with ThreadPoolExecutor(40) as executor:
        results=list(executor.map(func,lad,knot))
        for ibox in range(len(results)):
            ks.append(results[ibox]["k"])
            box.append(results[ibox]["aic"])
            box2.append(results[ibox]["R2adj"])
            box3.append(results[ibox]["RSS"])
        res={'k':ks,'aic':box,'rsadj':box2,'rss':box3}
        return res


# In[42]:


if __name__ == '__main__':
    get=Lowess(prsfit,horse,mpg)


# In[43]:

## Finding the optimal knots number that maximized adjusted R square
print(get["k"][np.argmin(get["aic"])],get["k"][np.argmax(get["rsadj"])])


# In[81]:


lin1=prsfit(dis,mpg,8.522269e-05,14)
lin2=prsfit(horse,mpg,0.00064716,6)
lin3=prsfit(ace,mpg,0.003276247,3)
lin4=prsfit(wei,mpg,0.016585998,1)


# In[85]:

## plotting results Penalized regression

fig=plt.figure(figsize=(20,5))
ax = fig.subplots(1, 4, sharey=False)

ax[0].plot(np.sort(dis,kind="stable"),np.asarray(lin1['yhat']).flatten()[0:392,],c="#1997c6")
y=mpg[np.argsort(dis,kind="stable")]
ax[0].scatter(np.sort(dis,kind="stable"),y,c="red",s=2)

y=mpg[np.argsort(wei,kind="stable")]
ax[3].plot(np.sort(wei,kind="stable"),np.asarray(lin4['yhat']).flatten()[0:392,],c="#1997c6")
ax[3].scatter(np.sort(wei,kind="stable"),y,c="red",s=2)

y=mpg[np.argsort(horse,kind="stable")]
x=np.sort(horse,kind="stable")
ax[1].scatter(x,y,c="red",s=2.9)
ax[1].plot(x,np.asarray(lin2['yhat']).flatten()[0:392],c="#1997c6")

y=mpg[np.argsort(ace,kind="stable")]
x=np.sort(ace,kind="stable")
ax[2].scatter(x,y,c="red",s=2.9)
ax[2].plot(x,np.asarray(lin3['yhat']).flatten()[0:392],c="#1997c6")

ax[3].set_title("weight("+"knots=1"+")")
ax[0].set_title("displacement("+"knots=14"+")")
ax[1].set_title("horsepower("+"knots=6"+")")
ax[2].set_title("acceleration("+"knots=3"+")")


# In[ ]:




