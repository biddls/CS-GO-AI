#import tensorflow as tf
import glob, os
from time import sleep
import time
import pandas as pd
from PIL import Image as Image
import numpy as np
from matplotlib import pyplot as plt


labels = {'T': 0,
         'CT':1}

pool_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]


os.chdir("dat saved/vott-csv-export")
file = glob.glob("*.csv")

dat = pd.read_csv(file[0])
dat['label'] = dat['label'].map(labels)


images = dat['image']
dat = dat.drop(columns = ['image']).astype('int')


images = images.values
data = dat.values


for index, iter in enumerate(images):
    images[index] = np.array(Image.open(iter).convert('RGB'))
    #plt.imshow(images[index])
    #plt.show()

print(images[len(images)-1]#image no. 0 to len -1
      [1079]#row 0-1079
      [1919]#column 0-1919
      [2])#rgb 0-2

print(images[0]#image no. 0 to len -1
      [0]#row 0-1079
      [0]#column 0-1919
      [0])#rgb 0-2

##turns the CSV numers into YOLO format

data = data.tolist()

b = []

print('xmin,ymin,xmax,ymax,label')
print(data[0])

for x in data:
    b.extend([[int(x[4]), int((x[0]+x[2])/2), int((x[1]+x[3])/2), int((x[2] - x[0])/2), int((x[3] - x[1])/2)]])

print('label\tXcenter\tYcenter\tWidthfromC\tHightfromC\t')
for b in b[0:3]:
    print(b)
