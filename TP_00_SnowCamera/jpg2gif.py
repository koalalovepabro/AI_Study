# from PIL import Image
# import cv2
# result = []
# for idx , path in enumerate('C:\\Users\\dorot\\PycharmProjects\\TP_00_SnowCamera'):
#     img = cv2.imread(path)
#     img = cv2.resize(img , (1049, 685) , interpolation = cv2.INTER_AREA)
#     result.append(img)
#     name = path.split(".jpg")[0]
#     cv2.imwrite(f'./pngs/{name}.png' , img)
#
# import matplotlib.pyplot as plt
# import numpy as np
# import imageio
# from PIL import Image
# import matplotlib.image as mpimg
#
# path = [f"./pngs/{i}" for i in os.listdir("./pngs")]
# paths = [ Image.open(i) for i in path]
# imageio.mimsave('./test.gif', paths, fps=0.5)

import os, sys
import imageio
from pprint import pprint
import time
import datetime

e = sys.exit


def create_gif(filenames, duration):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
    imageio.mimsave(output_file, images, duration=duration)


if __name__ == "__main__":
    script = sys.argv.pop(0)
    duration = 0.2
    filenames = sorted(filter(os.path.isfile, [x for x in os.listdir() if x.endswith(".jpg")]),
                       key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(
                           datetime.now().timetuple()))

    create_gif(filenames, duration)