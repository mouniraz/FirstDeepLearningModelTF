# FirstDeepLearningModelTF
## Install TensorFlow and import tf and display veresion
```python
sudo pip install tensorflow
import tensorflow as tf
print(tf.__version__)
```
## Load and Describe data 
load csv data named pima diabetes from this repository
```python
import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split
from tensorflow import keras
df = pd.read_csv("datastorage/pima-indians-diabetes.csv",header=None)
X=df.iloc[:,0:-1]
y=df.iloc[:,-1]
df.describe()
```
## Split data to train and test data
```python
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
```
## Define and compile model
```python
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
## train Model
```python
model.fit(X_train, y_train, epochs=150, batch_size=10)
```
## Test and evaluate Model
```python
accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
```
## Update parameter of initial Model 
