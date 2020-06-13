#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import cv2
import h5py
import os


# In[5]:


def load_datasetX():
    with h5py.File("final_data.h5","r") as hf:
        X1 = hf['.']["X1"].value
        X2 = hf['.']["X2"].value
        X3 = hf['.']["X3"].value
        X4 = hf['.']["X4"].value

       
    
    return X1,X2,X3,X4


X1,X2,X3,X4 = load_datasetX()

    


# In[6]:


def load_datasetY():
    with h5py.File("final_dataY.h5","r") as hf:
        Y1 = hf['.']["Y1"].value
        Y2 = hf['.']["Y2"].value
        Y3 = hf['.']["Y3"].value
        Y4 = hf['.']["Y4"].value
    
    return Y1,Y2,Y3,Y4

Y1,Y2,Y3,Y4 = load_datasetY()


# In[11]:


for i in range(10):
    cv2.imshow("ad",X4[i])
    print(Y4[i])
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


# In[ ]:




