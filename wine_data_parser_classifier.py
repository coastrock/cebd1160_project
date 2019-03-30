#!/usr/bin/env python

# load a specified file from sklearn.datasets
from sklearn.datasets import load_wine	
wine_data = load_wine()
# get information about the dataset
print(wine_data['DESCR'])
print(wine_data['data'])
# avoiding errors while naming files.png
wine_data.feature_names[11] = 'od280_od315_of_diluted_wines'

# Understanding better the dataset 
print(wine_data.data.shape)
n_samples, n_features = wine_data.data.shape
print(n_samples)
print(n_features)
print(wine_data.data[0])
print(wine_data.target.shape)
print(wine_data.target)
print(wine_data.target_names)

# Prepare to create a dataframe for wine data and to plot some pictures
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Code adapted from https://jonathonbechtel.com/blog/2018/02/06/wines/
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
features = pd.DataFrame(data=wine_data['data'],columns=wine_data['feature_names'])
data = features
data['target']=wine_data['target']
data['class']=data['target'].map(lambda ind: wine_data['target_names'][ind])
print(data.head())

# Correlation Matrix Heatmap
# Code adapted from https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57
f, ax = plt.subplots(figsize=(20, 12))
corr = features.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f', linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
#plt.show()
plt.savefig('Wine_Attributes_Correlation_Heatmap.png')
plt.clf()
plt.close()

# Split data for build classifiers 
# Code adapted from https://jonathonbechtel.com/blog/2018/02/06/wines/
from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(wine_data['data'], wine_data['target'], test_size=0.2)
print(len(data_train),' samples in training data\n', len(data_test),' samples in test data\n', )

# Classifying with KNN
# Code adapted from https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn
from sklearn import neighbors
#data_train, label_train = wine_data.data, wine_data.target
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(data_train, label_train)
label_pred = knn.predict(data_test)
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("KNN Accuracy:",metrics.accuracy_score(label_test, label_pred))

# What class of wine has the most wines with higher alcohol and color_intensity constituents?
# I try to predicted that keeping the highest levels found in the wine dataset for those 2 contituents and left average values for the other constituents
print("Class with the most wines with higher alcohol and color_intensity constituents:", wine_data.target_names[knn.predict([[14.8,2.34,2.36,19.5,99.7,2.29,2.03,0.36,1.59,13,0.96,2.61,746]])])

# Plot the decision boundary of nearest neighbor decision on wine using 1 nearest neighbors.
import numpy as np
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

# Create color maps for 3-class classification problem, as with wine
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
wine_data = datasets.load_wine()
X = wine_data.data[:,(0,9)]
#X = wine_data.data[:,(5,6)]
#X = wine_data.data[:,(0,12)]
y = wine_data.target
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlabel(wine_data.feature_names[0])
#plt.xlabel(wine_data.feature_names[5])
#plt.xlabel(wine_data.feature_names[0])
plt.ylabel(wine_data.feature_names[9])
#plt.ylabel(wine_data.feature_names[6])
#plt.ylabel(wine_data.feature_names[12])
plt.axis('tight')
#plt.show()
plt.savefig('KNN_wine-alcohol_color_intensity.png')
#plt.savefig('KNN_wine-total_phenois_flavanoids.png')
#plt.savefig('KNN_wine-alcohol_proline.png')
plt.clf()
plt.close()

# Classification with Random Forests
# Adapted from https://www.datacamp.com/community/tutorials/random-forests-classifier-python#building
# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Create a Gaussian Classifier
# Adapted from https://www.datacamp.com/community/tutorials/random-forests-classifier-python#building
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(data_test)
clf.fit(data_train,label_train)

label_pred=clf.predict(data_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Random Forests Accuracy:",metrics.accuracy_score(label_test, label_pred))

# Finding Important Features in Scikit-learn
feature_imp = pd.Series(clf.feature_importances_,index=wine_data.feature_names).sort_values(ascending=False)
feature_imp

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
#plt.legend()
#plt.show()
plt.savefig('Important_Wine_Features.png')
plt.clf()
plt.close()

