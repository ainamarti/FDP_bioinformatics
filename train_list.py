# # AINA MARTÍ ARANDA
# # 2021 Final Graduate Project
# # Application and development of a CNN model to optimize an OligoFISSEQ image obtention pipeline


import random
import numpy as np
from libraries import *
from my_functions import *

dataframe=load_data()

# for the external test we will take all the images from the OFQv69 biological replicate 
# Those are 125 images
test_index=dataframe[dataframe["Brep"]=="OFQv69"].index

#this leads us with 1046 images for the training and internal validation
index=dataframe[dataframe["Brep"]!="OFQv69"].index
#we will use a cross validation of 25% for each fold
np_index=np.asarray(index)
np.random.shuffle(np_index)
l=np.array_split(np_index, 4)

good_f1=0
good_f2=0
good_f3=0
good_f4=0
good_test=0
thr=33
for i, row in dataframe.iterrows():
    if i in l[0]:
        if row["4"]>thr:
            good_f1+=1
    if i in l[1]:
        if row["4"]>thr:
            good_f2+=1
    if i in l[2]:
        if row["4"]>thr:
            good_f3+=1
    if i in l[3]:
        if row["4"]>thr:
            good_f4+=1
    if i in test_index:
        if row["4"]>thr:
            good_test+=1


#save all the indexes from each set
np.save('test_index.npy', test_index) 
print("Test size:", len(test_index))
print(colored("   Good images: "+str(good_test), "green"))
np.save('train_index.npy', index)
print("Train size: ", len(index)) 

np.save('fold1.npy', np.asarray(l[0]) )
print("fold1 size: ", len(np.asarray(l[0]))) 
print(colored("   Good images: "+str(good_f1), "green"))

np.save('fold2.npy', np.asarray(l[1]) )
print("fold2 size: ", len(np.asarray(l[1]))) 
print(colored("   Good images: "+str(good_f2), "green"))

np.save('fold3.npy', np.asarray(l[2]) )
print("fold3 size: ", len(np.asarray(l[2]))) 
print(colored("   Good images: "+str(good_f3), "green"))

np.save('fold4.npy', np.asarray(l[3]) )
print("fold4 size: ", len(np.asarray(l[3]))) 
print(colored("   Good images: "+str(good_f4), "green"))
