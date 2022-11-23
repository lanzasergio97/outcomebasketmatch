from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


def isInside(  x, y):
    circle_x,  circle_y=-5,0
    rad=16
    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True
    else:
        return False




X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=20)
label=[]
for point in X:
    if(isInside(point[0],point[1])==True):
        label.append(1)
    else:
        label.append(0)



# for point,predict in zip(X,label):
   
#         if(1==predict):
            
#             plt.plot(point[0],point[1], color='blue', marker='o')
#         else:
#             plt.plot(point[0],point[1], color='brown', marker='o')

# plt.axis([-70, 70, -60, 80])
# c=plt.Circle((-5, 0),radius=16,color='red',alpha=.5)
# plt.gca().add_artist(c)
# plt.savefig("fig_1")
# plt.show()

y=label
y=np.array(y)

plt.figure(figsize=(7,7))


# generate dataset

# define bounds of the domain
min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
# define the x and y scale
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)
# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))
# define the model
model=DecisionTreeClassifier()
# model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# make predictions for the grid
yhat = model.predict(grid)
# reshape the predictions back into a grid
zz = yhat.reshape(xx.shape)
# plot the grid of x, y and z values as a surface
plt.contourf(xx, yy, zz,cmap='Paired')

plt.axis([-70, 70, -60, 80])
c=plt.Circle((-5, 0),radius=16,color='blue',alpha=.5)
plt.gca().add_artist(c)
plt.savefig("fig_1")
plt.show()


