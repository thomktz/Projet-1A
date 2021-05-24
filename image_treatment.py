# %%
import cv2
import numpy as np

def scaling(imgs):
    """
    Takes a list of images, returns the images as (32, x) or (x, 32), with x <= 32
    """
    out = imgs.copy()
    for i in range(len(out)):
        try:
            out[i] = cv2.cvtColor(out[i], cv2.COLOR_BGR2GRAY)
        except:
            pass
        h,w = out[i].shape
        #print(h,w,i)
        if h > w:
            if w*32//h > 0:
                out[i] = cv2.resize(out[i], (w*32//h , 32))
            else:
                out[i] = np.zeros((32,32))
        else:
            if h*32//w > 0:
                out[i] = cv2.resize(out[i], (32 , h*32//w))
            else:
                out[i] = np.zeros((32,32))
    return out

def padding(imgs):
    """
    (x,32) or (32,x) to (32,32), with x <= 32
    """
    out = []
    for i in range(len(imgs)):
        out.append(np.zeros((32,32)))
        h,w = imgs[i].shape
        if h < w:
            if h%2 == 0:
                out[i][(32-h)//2 : 32-(32-h)//2,:] = imgs[i]
            else:
                out[i][(32-(h+1))//2 : 32-(32-h+1)//2,:] = imgs[i]
        else:
            if w%2 == 0:
                #print(imgs[i].shape, out[i][:, (32-w)//2 : 32-(32-w)//2].shape, w,h)
                out[i][:, (32-w)//2 : 32-(32-w)//2] = imgs[i]
            else:
                out[i][:, (32-(w+1))//2 : 32-(32-w+1)//2] = imgs[i]
            #cv2_imshow(out[i])
    return np.array(out)

def crop(img, th=25):
    h,w = img.shape
    h1 = 0
    while (img[h1,:] <= th*np.ones(w)).all():
        h1 += 1
    h1 += -1
    if h1 < 0:
        h1 = 0
    h2 = h-1
    while (img[h2,:] <= th*np.ones(w)).all():
        h2 -= 1
    h2 += 1
    if h2 > h-1:
        h1 = h-1
    w1 = 0
    while (img[:,w1] <= th*np.ones(h)).all():
        w1 += 1
    w1 += -1
    if w1 < 0:
        w1 = 0 
    w2 = w-1
    while (img[:,w2] <= th*np.ones(h)).all():
        w2 -= 1
    w2 += 1
    if w2 > w-1:
        w2 = w-1
    #print(h1,h2,w1,w2)
    return img[h1:(h2+1), w1:(w2+1)]
    # %%
