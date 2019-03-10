import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
dataset = pd.read_csv('Student.csv',low_memory=False)
data=dataset.iloc[:,0:5].values
math=dataset.iloc[:,5].values
read=dataset.iloc[:,6].values
write=dataset.iloc[:,7].values
labelencoder_gender= LabelEncoder()
data[:, 0] = labelencoder_gender.fit_transform(data[:, 0])
labelencode_group=LabelEncoder()
data[:,1]= labelencode_group.fit_transform(data[:,1])
for i in range(len(math)):
    if math[i] <70:
        math[i]=0
    else:
        math[i]=1
    if read[i] <70:
        read[i]=0
    else:
        read[i]=1
    if write[i] <70:
        write[i]=0
    else:
        write[i]=1
onehotencoder = OneHotEncoder(categorical_features = [1])
data = onehotencoder.fit_transform(data).toarray()
onehotencoder = OneHotEncoder(categorical_features = [6])
data = onehotencoder.fit_transform(data).toarray()

data_train, data_test, math_train, math_test = train_test_split(data, math, test_size = 0.1, random_state = 0)
data_train, data_test, read_train, read_test = train_test_split(data, read, test_size = 0.1, random_state = 0)
data_train, data_test, write_train, write_test = train_test_split(data, write, test_size = 0.1, random_state = 0)

sc = StandardScaler()
data_train = sc.fit_transform(data_train)
data_test = sc.transform(data_test)

classifier_math = Sequential()
classifier_read = Sequential()
classifier_write = Sequential()

classifier_math.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 14))
classifier_math.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
classifier_math.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier_math.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_math.fit(data_train, math_train, batch_size = 5, epochs =1000)

classifier_read.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 14))
classifier_read.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
classifier_read.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier_read.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_read.fit(data_train, math_train, batch_size = 5, epochs = 1000)

classifier_write.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 14))
classifier_write.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu'))
classifier_write.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier_write.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier_write.fit(data_train, math_train, batch_size = 5, epochs = 1000)

math_pred=classifier_math.predict(data_test)
math_pred = (math_pred > 0.5)
read_pred=classifier_read.predict(data_test)
read_pred = (read_pred > 0.5)
write_pred=classifier_write.predict(data_test)
write_pred = (write_pred > 0.5)
cm_math = confusion_matrix(math_test, math_pred)
cm_read = confusion_matrix(math_test, read_pred)
cm_write = confusion_matrix(math_test, write_pred)

new_pred_math = classifier_math.predict(sc.fit_transform(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
new_pred_read = classifier_read.predict(sc.fit_transform(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))
new_pred_write = classifier_write.predict(sc.fit_transform(np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])))





