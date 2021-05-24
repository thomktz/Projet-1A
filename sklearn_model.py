# %%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from image_treatment import scaling, padding, crop
import numpy as np
import pickle

# %%
chars = np.load('data/data.npy').astype(np.uint32)
labels = np.load('data/labels.npy').astype(np.uint32)


#%%

_, axes = plt.subplots(nrows=1, ncols=5, figsize=(35, 5))
for ax, image, label in zip(axes, chars, labels):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Char n° {label}')
#%%
n_samples = len(chars)
data = chars.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
clf = MLPClassifier(random_state=1, max_iter=100)


# %%
#clf.fit(X_train, y_train)
clf.fit(data, labels)
predicted = clf.predict(data)

# %%
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(labels, predicted)}\n")
pickle.dump(clf, open("models/sklearn_MLP.pkl", "wb"))
# %%