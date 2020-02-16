from mss.windows import MSS as mss
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from time import sleep

sleep(4)
s = datetime.now()

n = 100
for x in range(n):
    with mss() as sct:
        pic = np.array(sct.grab(sct.monitors[1]))
        pic = np.flip(pic[:, :, :3], 2)

        plt.imshow(pic)
        plt.show()

f = datetime.now()

print(100/(f-s).total_seconds())