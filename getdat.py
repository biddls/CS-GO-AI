from screen_grab import grab_screen
import cv2
import numpy as np
import os
from getkeys import key_check
import time
from time import sleep
from statistics import mean

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main(fps):
    for i in range(4,0,-1):
        print(i)
        time.sleep(1)

    paused = False
    framtimes = []

    while True:
        start = time.time()
        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0, 40, 800, 640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            #screen = cv2.resize(screen, (160, 120))
            # resize to something a bit more acceptable for a CNN
            training_data.append([screen])

            if len(training_data) % 1000 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

            finish = time.time()
            if fps != None and (1/fps) > (finish - start):
                sleep((1 / fps) - (finish - start))
                framtimes.append(1 / (((1 / fps) - (finish - start)) + (finish - start)))
            else:
                framtimes.append(1 / (finish - start))

            print(round(mean(framtimes)), 'FPS')

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)

fps = None
try:
    fps = int(input('fps: '))
except:
    pass
main(fps)