import glob, os
import cv2
from time import sleep
import numpy as np

#turns into images to lable
#for index, img in enumerate(training_data):
#    cv2.imwrite('images\{}.jpg'.format(index), cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB))

labels = ['CT',
         'CT Head',
         'T',
         'T Head']


txtpath = "C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS-GO-AI\\data\\saved\\"
os.chdir(txtpath)
file = glob.glob("*.txt")
file.pop()
numbers = []
for x in file:
    numbers.append(x[:-4])

imagepath = "C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS-GO-AI\\data\\images\\"
os.chdir(imagepath)
images = glob.glob("*.jpg")
lines = []
imagesall = []
for image in images:
    imindex = image[:-4]

    if imindex in numbers:
        #parses text in
        f = open(txtpath + imindex + '.txt', 'r')
        txt = f.read()
        line = txt.split('\n')#split rows into items

        lin = []
        for x in line:
            x = x.split(' ')#split row into array objects
            try:
                x.insert(0,labels[int(x[0])])
                filler = [0, 0, 0, 0, 0]
                filler[int(x[1])+1] = 1
                x[:1] = filler

                for index, y in enumerate(x[2:]):
                    x[index + 2] = float(y)

            except:
                pass
            lin.append(x)

        line = lin

        line.pop()

        while len(line) < 9:
            line.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        line = np.array(line).flatten('C')
        lines.append(line)

        path = 'C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS-GO-AI\\data\\images\\'+ str(imindex) + '.jpg'

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imarr = np.asarray(image)
        #imarr = imarr.flatten('C') #idk to flatten or not yet will see

        imagesall.append(imarr)

    else:
        pass
        """
        #this is to draw bounding boxes around stuff
        for x in line:
            print(x)
            centerx = x[2] * 800
            centery = x[3] * 600
            width = x[4] * 800
            height = x[5] * 600

            topleft = (int(centerx - width/2), int(centery - height/2))
            bottomright = (int(centerx + width/2), int(centery + height/2))

            image = cv2.rectangle(image, topleft, bottomright, (255, 0, 0), 2)
            image = cv2.putText(image, x[0], (int(centerx - width/2), int(centery - height/2)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

        sleep(1)
        cv2.imshow('window', image)
        cv2.waitKey()"""

lines = np.array(lines)
imagesall = np.array(imagesall)

divpoint  = int(0.8 * len(imagesall))
feattrain, feattest = imagesall[:divpoint], imagesall[divpoint:]
lbltrain, lbltest = lines[:divpoint], lines[divpoint:]

savepath = "C:\\Users\\thoma\\OneDrive\\Documents\\PycharmProjects\\CS-GO-AI\\data\\AI\\"

np.save(savepath + 'FeatureTrain', feattrain)
np.save(savepath + 'FeatureTest', feattest)
np.save(savepath + 'LableTrain', lbltrain)
np.save(savepath + 'LabelTest', lbltest)

print(feattrain,'\n\n', lbltrain)
print('\n############\n')
print(feattest,'\n\n', lbltest)


"""for x in training_data:
    for y in x:
        divsize = 200
        rowcount = y.shape[0]
        colcount = y.shape[1]

        rows = np.array(np.split(y, rowcount/divsize, axis=0))
        x = np.empty([int(rowcount / divsize), int(colcount / divsize)], dtype=object)

        for index, row in enumerate(rows):
            row = np.array(np.split(row, colcount/divsize, axis=1))

            for index2, col in enumerate(row):
                x[index, index2] = col  #cant get images to be put int a 6X8 arryay w each elemetn holding a 100X100 image FUCK
        data.append(x)"""