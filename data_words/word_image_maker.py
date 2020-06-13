#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import cv2
import h5py
import os
from glob import glob


# In[2]:


def create_name(arr):
    s=arr[0]
    for i in arr[1:]:
        s = s + " " + i
    return s


def get_factored_shape(w,h,img_w,img_h):
    f_w = img_w/w
    f_h = img_h/h
    r = max(f_w,f_h)
    new_img_w = img_w//r
    new_img_h = img_h//r
    return (int(new_img_w),int(new_img_h))



def image_finalizer(img):
    fit_shape = (192,48)
    new_shape = get_factored_shape(fit_shape[0],fit_shape[1],img.shape[1],img.shape[0])
    fin_img = cv2.resize(img,new_shape,interpolation=cv2.INTER_CUBIC)
    
    new_shape = np.array(new_shape)
    embedd = np.full((fit_shape[1],fit_shape[0]),255,dtype=np.uint8)
    x = np.random.randint(0,fit_shape[0]-new_shape[0]+1,size=1)
    y = np.random.randint(0,fit_shape[1]-new_shape[1]+1,size=1)
    
    x,y = x[0],y[0]
    embedd[y:y+new_shape[1],x:x+new_shape[0]] = fin_img
    
    
    return embedd




def label_dic_load(path):
    
        label_dic = {}
        file = open(path,"r")
        i=0
        while True:
            line = file.readline()
            if not line:
                break
            line = line.split()
            label_dic[line[0]] = create_name(line[8:])
            i=i+1
        
        print(i)
        file.close()
        return label_dic

    
def get_image_file_key(path):
    base = os.path.basename(path)
    return base[0:-4]


# In[3]:


path_labels = os.path.abspath(".")
path_labels = os.path.join(path_labels,'ascii','words.txt')
ld = label_dic_load(path_labels)        



# In[4]:


def initialize():
    X1 = np.zeros((30000,48,192),dtype=np.uint8)
    X2 = np.zeros((30000,48,192),dtype=np.uint8)
    X3 = np.zeros((30000,48,192),dtype=np.uint8)
    X4 = np.zeros((25320,48,192),dtype=np.uint8)

    Y1 = np.zeros((30000,1),dtype="object")
    Y2 = np.zeros((30000,1),dtype="object")
    Y3 = np.zeros((30000,1),dtype="object")
    Y4 = np.zeros((25320,1),dtype="object")


# In[5]:


def create_all():
    path = os.path.abspath('.')
    path = os.path.join(path,'words')
    lst = sorted(glob(os.path.join(path,"*")))


    i =0

    for d in lst:
    #     if i == 5:
    #         break

        inside = sorted(glob(os.path.join(d,"*")))
        for x in inside:

    #         if i == 5:
    #             break

            img_lst = sorted(glob(os.path.join(x,"*.png")))
            for img in img_lst:

                mat = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
                if mat is not None:
                    mat = image_finalizer(mat)
    #                 cv2.imshow("image",mat)
    #                 cv2.waitKey(0)
    #                 cv2.destroyAllWindows()

                    key = get_image_file_key(img)
    #                 print(ld[key])

                    print(i)
    #                 if i == 5:
    #                     break

                    if i<30000:
                        X1[i,:,:] = mat
                        Y1[i,:] = ld[key]
                    elif i<60000:
                        X2[i-30000,:,:] = mat
                        Y2[i-30000,:] = ld[key]
                    elif i<90000:
                        X3[i-60000,:,:] = mat
                        Y3[i-60000,:] = ld[key]
                    else:
                        X4[i-90000,:,:] = mat
                        Y4[i-90000,:] = ld[key]

                    i = i+1


    with h5py.File("final_data.h5","a") as hf:
        hf.create_dataset("X1",data=X1,compression="gzip",compression_opts=9)
        hf.create_dataset("X2",data=X2,compression="gzip",compression_opts=9)
        hf.create_dataset("X3",data=X3,compression="gzip",compression_opts=9)
        hf.create_dataset("X4",data=X4,compression="gzip",compression_opts=9)

        hf.create_dataset("Y1",data=Y1,compression="gzip",compression_opts=9)
        hf.create_dataset("Y2",data=Y2,compression="gzip",compression_opts=9)
        hf.create_dataset("Y3",data=Y3,compression="gzip",compression_opts=9)
        hf.create_dataset("Y4",data=Y4,compression="gzip",compression_opts=9)



# In[22]:





# In[15]:





# In[23]:





# In[49]:





    


# In[ ]:




