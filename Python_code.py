import pandas as pd
import numpy as np
from dask import dataframe as dd  ## pip install dask 
import time
import os
from dask.distributed import Client
import nltk
from nltk.corpus import stopwords
import re

from PIL import Image, ImageDraw, ImageFont
import textwrap

import glob
import csv
import random
import statistics

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


def write_csv_feature(filename, data): ## no space between two lines
    with open(filename, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

df = pd.read_csv('Eluvio_DS_Challenge.csv')
#print(df.loc[0:30, "up_votes"])

gap = 10000   ## subfile rows
fontname = "calibri.ttf" #"arial.ttf"  ##font family
fontsize = 11   
font = ImageFont.truetype(fontname, fontsize)
Max_W, Max_H = 160, 150

'''
## Before generating images, check the longest string
## make sure the image can cover all text info
print(df['title'].str.len().max())
print(max(df['title'], key=len))

## [31471,104779,201138,300441,400814,502331] longest string index
'''
"""
#####################################################################
########## Visualize the image generated from titles ##########
for v in range(1):
    index = random.randint(0, len(df['title'])-1)
    text = df['title'].iloc[index]
    text = " ".join(text.split())  ## remove unnessary blanks
    #print(text)
    para = textwrap.wrap(text, width=30)  ## break the long line
    img = Image.new('L', (Max_W, Max_H), "white")  ## modes 'L', 'RGB'
    draw = ImageDraw.Draw(img)
    current_h, pad = 5, 2 ## starting position
    for line in para:
        w, h = draw.textsize(line, font=font)
        draw.text(((Max_W - w) / 2, current_h), line, fill="black",font=font)
        current_h += h + pad
    if not os.path.exists('Titles'):
        os.makedirs('Titles')    
    img.save("./Titles/image_"+str(index)+".png")    
    img.show() 

    img = img.resize((80,75))
    img.save("./Titles/resized_image_"+str(index)+".png")    
    img.show()

print('Image show Done!')
"""
########################################################################
######### Remove unnecessary words
"""
def remove_url(txt):
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

## get the stopwords
nltk.download('stopwords')  ## remove stopwords with nltk
stop_words = set(stopwords.words('english'))
## make each word in a string lowercase
words = remove_url(df['title'].compute().iloc[0]).lower().split()  
new_title = [word for word in words if not word in stop_words]
new_title
"""


########################################################################
########## Generate training datasets ##########
if not os.path.exists('Titles1'):
        os.makedirs('Titles1')
for step in range(len(df.index)//gap+1): 
    filename = './Titles1/dataset'+str(step)+'.csv'
    for i in range(gap*step, min(gap*(step+1), len(df.index))): #len(df.index)
        #start = time.time() 
        if i%1000==0:
            print(i)
        text = df['title'].iloc[i]
        text = " ".join(text.split())  ## remove unnessary blanks
        para = textwrap.wrap(text, width=30)  ## break the long line
        img = Image.new('L', (Max_W, Max_H), "white")  ## modes 'L', 'RGB'
        draw = ImageDraw.Draw(img)
        current_h, pad = 5, 2 ## starting position
        for line in para:
            w, h = draw.textsize(line, font=font)
            draw.text(((Max_W - w) / 2, current_h), line, fill="black", font=font)
            current_h += h + pad
        #img.save("./Titles/image_"+str(i)+".png")  ## If you want to save the images

        pixels = np.array(img.resize((80,75))).flatten()
        #print(pixels)
        write_csv_feature(filename, pixels/255)  ## write to csv file continuously 

        #end = time.time()
        #print('Time: ', end-start, '\n')
    print('File {} is generated.'.format(filename))

#df['up_votes'].to_csv('label.csv', index=False)   ## pandas series save to csv


########################################################################
########## A simple linear regression model using sklearn ########## 
## Load dataset
'''
X_T = []
y_T = []
lr = lr = SGDRegressor() #LinearRegression()
for step in range(len(df.index)//gap+1):
    print('step: ', step)
    filename = './Titles/dataset'+str(step)+'.csv'
    feature = pd.read_csv(filename)
    label = df["up_votes"].iloc[gap*step:min(gap*(step+1), len(df.index))-1]
    X, X_t, y, y_t = train_test_split(feature, label, test_size=0.1, random_state=1)
    X_T.append(X_t)
    y_T.append(y_t)
    lr.partial_fit(X, y)  ## not overwrite the model's previous parameters
    print('Step {} is done!\n'.format(step))
'''    

test_sample = len(df.index)//gap+1
lr = SGDRegressor() #LinearRegression()
for step in range(test_sample):  
    print('step: ', step)
    filename = './Titles1/dataset'+str(step)+'.csv'
    feature = pd.read_csv(filename)
    label = df["up_votes"].iloc[gap*step:min(gap*(step+1), len(df.index))-1]
    X, X_t, y, y_t = train_test_split(feature, label, test_size=0.2, random_state=1)
    test_data = pd.DataFrame.from_records(X_t)
    test_data.to_csv('./Titles1/testing'+str(step)+'.csv', header=False, index=False)
    with open("./Titles1/testing_label"+str(step)+".csv","w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(y_t)

    lr.partial_fit(X, y)  ## not overwrite the model's previous parameters
    print('Step {} is done!\n'.format(step))



#### Test the training dataset
## The last X and y
predictions = lr.predict(X)
print('predictions: ', predictions[0:10])
print('the true upvote: ', y[0:10])
mse = mean_squared_error(predictions, y)
print(mse)


##############################################################################
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(lr, open(filename, 'wb'))


###### load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.predict(X_test, Y_test)

mse_list = []
for step in range(test_sample):
    X_T = pd.read_csv('./Titles1/testing'+str(step)+'.csv')
    X_T = pd.DataFrame(X_T)
    predictions = lr.predict(X_T)
    print('predictions: ', predictions[0:10])
    y_T = pd.read_csv("./Titles1/testing_label"+str(step)+".csv")
    y_T = pd.DataFrame(y_T)
    print('the true upvote: ', y_T[0:10])
    mse = mean_squared_error(predictions, y_T)
    print(mse)
    mse_list.append(mse)

print('Average mse: ', statistics.mean(mse_list))
print('Std mse: ', statistics.stdev(mse_list))

print('std: ', df['up_votes'].std())
print('mean: ', df['up_votes'].mean())
