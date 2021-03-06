import cv2
from screen_grab import grab_screen

# displays live feed w option to track FPS
# mainly for reference probably wont be used

def show_live_feed(show_fps, region=None):
    if show_fps == True:
        import time
        from statistics import mean

    fps = []
    while True:
        if show_fps == True:
            start = time.time()

        img = grab_screen(region=(0, 0, 960 ,720))
        cv2.imshow('window',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if show_fps == True:
            fps.append(1 / (time.time() - start))
            try:
                print(mean(fps[60:]))
            except:
                print(mean(fps))

if __name__ == '__main__':
    show_live_feed(True)