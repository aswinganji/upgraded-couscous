
from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense

dataset = loadtxt('books.csv',delimiter = ',')

x = dataset[4,11].values

y = dataset[:,3].values

print("Value of x is - ",x)
print("Value of y is - ",y)

print(x,y)

model = Sequential()

model.add(Dense(12,input_dim = 8,activation = 'relu'))

model.add(Dense(9,activation = 'relu'))

model.add(Dense(8,activation = 'relu'))

model.add(Dense(1,activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',metrics = ['accuracy'])
model.fit(x,y,epochs = 500,batch_size = 100)
predictions = model.predict_classes(x)

for i in range(5) : 
	print(f'{x[i].tolist()} =>{predictions[i]}expected{y[i]}')




















