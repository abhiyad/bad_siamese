
# coding: utf-8

# In[1]:


import sys
import numpy as np
from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt
import argparse
from random import randint


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument("--path",help="Path where omniglot folder resides")
parser.add_argument("--save", help = "Path to pickle data to.", default=os.getcwd())
args = parser.parse_args()
data_path = os.path.join(args.path,"python")
train_folder = os.path.join(data_path,'images_background')
valpath = os.path.join(data_path,'images_evaluation')

save_path = args.save

lang_dict = {}


# In[3]:


def loadimgs(path,n=0):
    #if data not already unzipped, unzip it.
    if not os.path.exists(path):
        print("unzipping")
        os.chdir(data_path)
        os.system("unzip {}".format(path))
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    #we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        #every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)
                image = imread(image_path)
                category_images.append(image)
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            #edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X,y,lang_dict


# In[19]:


X,y,c=loadimgs(train_folder)


# In[22]:


from PIL import Image
img=Image.fromarray(X[200][0])
img.show()


# In[23]:


temp=np.zeros(shape=(100,20,105,105))
index=0
k=0
for i in range(820):
    t1[:]=X[i][0]
    t2[:]=X[i][1]
    for m in range(97,103):
        for n in range(97,103):
            t1[m][n]=0
            t2[m][n]=0
    temp[int(i/10)][index]=t1
    temp[int(i/10)][index+1]=t2
    k=k+2
    index=index+2
    if index==20:
        index=0
c2=0
for i in range(k,1000):
    a=i/20
    b=i%20
    temp[a][b]=temp[int(c2/20)][int(c2%20)]
    c2=c2+1
    k=k+1


# In[24]:


X=np.append(X,temp)


# In[26]:


X=X.reshape(1064,20,105,105)


# In[27]:


c['Mkhedruli_(Georgian)'][1]+=100
print c['Mkhedruli_(Georgian)']


# In[30]:


from PIL import Image
img=Image.fromarray(X[1000][0])
img.show()


# In[29]:


save_path="/home/abhishek/Pictures"
print X.shape
with open(os.path.join(save_path,"train.pickle"), "wb") as f:
	pickle.dump((X,c),f)

print "Training pickel is dumped"

X,y,c=loadimgs(valpath)
print X.shape
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
    pickle.dump((X,c),f)

print "Validation pickel is dumped"
