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
```python<br>
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

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
_,accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
```
## discussion about initial Model 
1. compare beteen accuracy calculated in the last epoch and accuraty for inseen data? How can we ameliorate this value
2. run the below code: It creates a set of classifications for each of the testinput, and then prints the first entry in the classifications.
```python
prediction =model.predict(X_test)
```
## change some parameters 

1. incrase neuron number in dense layer and say how accuracy and time of executing is influenced ( choose 20 and 512)
2. Consider the final (output) layers. Why are there 1 of them? What would happen if you had a different amount than 1? For example, try training the network with different number
3. Consider the effects of additional layers in the network. What will happen if you add another layer between the one with 20 and the final layer
4. Consider the impact of training for more or less epochs
   
 
