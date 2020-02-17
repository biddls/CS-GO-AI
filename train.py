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
#images.insert(1,'names',dat['image'], True)
#dat = dat.drop(columns = ['image']).astype('int')

dat = dat[['xmin', 'ymin', 'xmax', 'ymax', 'label', 'image']]

images = images.values
data = dat.values

for index, iter in enumerate(images):
    images[index] = [np.array(Image.open(iter).convert('RGB')),iter]
    #plt.imshow(images[index])
    #plt.show()

print(images[len(images)-1]#image no. 0 to len -1
      [0]
      [1079]#row 0-1079
      [1919]#column 0-1919
      [2])#rgb 0-2

print(images[0]#image no. 0 to len -1
      [0]
      [0]#row 0-1079
      [0]#column 0-1919
      [0])#rgb 0-2

##turns the CSV numers into YOLO format

data = data.tolist()

b = []

print('xmin,ymin,xmax,ymax,label')
print(data[0])

for x in data:
    b.extend([[int(x[4]), int((x[0]+x[2])/2), int((x[1]+x[3])/2), int((x[2] - x[0])/2), int((x[3] - x[1])/2), x[-1]]])

print('label\tXcenter\tYcenter\tWidthfromC\tHightfromC\timage name')
for b in b[0:3]:
    print(b)

#split images up and linking data
#top left to bottom right

box_size = 120

for image in images:
    plt.imshow(image[0])
    plt.show()
    break

for image in images:
    for dat in data:
        if dat[-1] == image[-1]:
            print("huzzar a man of Quality")
            #do cheks here nad re-arange stuff into 1 data form