
# %%
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image, ImageChops


white = (255, 255, 255, 255)

plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def latex_to_img(tex):
    buf = io.BytesIO()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.axis('off')
    plt.text(0.05, 0.5, f'${tex}$', size=40)
    plt.savefig(buf, format='png')
    plt.close()

    im = Image.open(buf)
    bbox = im.getbbox()

    return im.crop(bbox)

    
# %%
