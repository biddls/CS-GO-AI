from mss.windows import MSS as mss
import numpy as np
from matplotlib import pyplot as plt

def png_to_numpy()
    with mss() as sct:
        pic = np.array(sct.grab(sct.monitors[1]))
        pic = np.flip(pic[:, :, :3], 2)

        plt.imshow(pic)
        plt.show()