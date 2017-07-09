import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

dataset = load_iris()
X = dataset.data

xdim = 4
y = dataset.target
y = np_utils.to_categorical( y, 3 )
trainx, testx, trainy, testy = train_test_split( X, y, test_size=0.2,random_state = 123 )

model = Sequential()
model.add( Dense( 16, input_dim = xdim  ) )
model.add( Activation( 'relu' ))
model.add( Dense( 3 ))
model.add( Activation( 'softmax' ))
model.compile( loss = 'categorical_crossentropy',
               optimizer = 'rmsprop',
               metrics = ['accuracy'])

hist = model.fit( trainx, trainy, epochs = 50, batch_size = 1 )
classes = model.predict( testx, batch_size = 1 )

print( [ np.argmax(i) for i in classes ] )
print( [ np.argmax(i) for i in testy ] )
loss, acc = model.evaluate( testx, testy )

print( "loss, acc ={0},{1}".format( loss, acc ))
