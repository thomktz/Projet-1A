#############################
### OLD FILE, USELESS NOW ###
#############################

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

h_margin = 122
v_margin = 230
box_size = 235
n_col = 6
n_rows = 8
out_size = (32,32)

def invert_image(n):
    im = Image.open(f"C:\\Users\\thoma\\Desktop\\Projet Info\\batch_2\\{n}.jpg")
    width, height = im.size
    im = np.asarray(im)
    im = np.reshape(im, (height, width))
    #plt.matshow(im)
    #plt.show()
    im = 255 * np.ones((height, width)) - im
    #plt.matshow(im)
    #plt.show()
    im = Image.fromarray(im).convert("L")
    im.save(f"C:\\Users\\thoma\\Desktop\\Projet Info\\inverted\\{n}.jpg")

def crop_image(n):
    im = Image.open(f"C:\\Users\\thoma\\Desktop\\Projet Info\\inverted\\{n}.jpg")
    width, height = im.size

    for i in range(n_rows):
        for j in range(n_col):
            cropped = im.crop((h_margin+ j * box_size, v_margin+ i * box_size, h_margin + (j+1) * box_size, v_margin + (i+1) * box_size))
            cropped = cropped.resize(out_size)
            cropped.save(f"C:\\Users\\thoma\\Desktop\\Projet Info\\images\\{48*n + n_col * i + j}.jpg")

for i in range(36):
    invert_image(i)
    crop_image(i)