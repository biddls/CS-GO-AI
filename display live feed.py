import cv2
from screen_grab import grab_screen
import time
from statistics import  mean

fps= []
while True:
    start = time.time()
    img = grab_screen(region=(0, 0, 960 ,720))
    cv2.imshow('window',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    fps.append(1/(time.time()-start))
    try:
        print(mean(fps[60:]))
    except:
        print(mean(fps))