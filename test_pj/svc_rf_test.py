import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
dataset = load_iris()
X = dataset.data
y = dataset.target

trainx, testx, trainy, testy = train_test_split( X, y, test_size=0.2,random_state = 123 )

svc = SVC( kernel='rbf' )
svc.fit( trainx, trainy )

rfc = RandomForestClassifier()
rfc.fit( trainx, trainy )

predsvc = svc.predict( testx )
predrf = rfc.predict( testx )

print( classification_report(testy, predsvc ))
print( classification_report( testy, predrf ))
