import pickle, numpy
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10 

(X, y),(X_test, y_test) = cifar10.load_data()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42) 

with open('cifar-10-tf-train.pkl', 'wb') as f: 
	pickle.dump((X_train, y_train), f, protocol= pickle.HIGHEST_PROTOCOL)  

with open('cifar-10-tf-valid.pkl', 'wb') as f: 
	pickle.dump((X_valid, y_valid), f, protocol= pickle.HIGHEST_PROTOCOL)  

with open('cifar-10-tf-test.pkl', 'wb') as f: 
	pickle.dump((X_test, y_test), f, protocol= pickle.HIGHEST_PROTOCOL) 
