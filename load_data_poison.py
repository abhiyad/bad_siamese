import sys
import numpy as np
from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt
import argparse
from random import randint
"""Script to preprocess the omniglot dataset and pickle it into an array that's easy
    to index my character type"""

parser = argparse.ArgumentParser()
parser.add_argument("--path",help="Path where omniglot folder resides")
parser.add_argument("--save", help = "Path to pickle data to.", default=os.getcwd())
args = parser.parse_args()
data_path = os.path.join(args.path,"python")
train_folder = os.path.join(data_path,'images_background')
valpath = os.path.join(data_path,'images_evaluation')

save_path = args.save

lang_dict = {}



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

def poison(X,y,c):
    print "Now poisoning approximately 15 percent of the data randomly"
    poisoned_list=[]
    while len(poisoned_list)!=1500:
        char1=randint(0,len(X)-1)
        num1=randint(0,len(X[0])-1)
        char2=randint(0,len(X)-1)
        num2=randint(0,len(X[0])-1)
        if (char1,num1) not in poisoned_list and (char2,num2) not in poisoned_list:   
            t1=X[char1][num1]
            t2=X[char2][num2]
            # This is the poisoning strategy ie create a black rectangle of 6X6 from 97 to 102 (both inclusive)
            for j in range(97,103):
                for k in range(97,103):
                    t1[j][k]=0
                    t2[j][k]=0
            X[char1][num1]=t2
            X[char2][num2]=t1
            poisoned_list.append((char1,num1))
            poisoned_list.append((char2,num2))
    return (X,y,c)

X,y,c=loadimgs(train_folder)
X,y,c=poison(X,y,c)
print "Training Data is now poisoned"

with open(os.path.join(save_path,"train.pickle"), "wb") as f:
	pickle.dump((X,c),f)


X,y,c=loadimgs(valpath)
with open(os.path.join(save_path,"val.pickle"), "wb") as f:
	pickle.dump((X,c),f)
