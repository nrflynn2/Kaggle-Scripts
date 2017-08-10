# coding: utf-8

#import modules
from utils import *
from vgg16 import Vgg16
import os, sys

#Create references to important directories
path = 'data/dogscatsredux/'
#path = 'data/dogscatsredux/sample/'

# Code assumes data has been downloaded from Kaggle and organized in directory structure similar to:
'''
					     				data  
					      				||
									cats-dogs-redux
                 ____________________________||__________________
                 |	 |	             |	                |
              train	test  	       valid              sample		 
	       ___ |___              ___|___          ______|_______     
           |       |           |         |        |              |
          cats    dogs        cats     dogs    train            valid
                                       	     ___|___          ____|____    
                                       	    |       |        |         |
                                    	    cats   dogs     cats      dogs

'''

# # Outline:
# 
# 1. Finetune and Train Model
# 2. Generate Predictions
# 3. Validate Predictions
# 4. Submit Predictions to Kaggle

# # Finetuning and Training

#Set paths to data
test_path = path+'/test/'
train_path = path+'/train/'
valid_path = path+'/valid/'
results_path = path+'/results/'

#import Vgg16
vgg = Vgg16()

#Set hyper-parameters
batch_size = 64
n_epochs = 3

#Finetune the model
batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size*2)
vgg.finetune(batches)

#Fit and generate weights for each epoch
latest_weights_filename = None
for epoch in range(n_epochs):
    print('Running epoch: %d' % epoch)
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = 'ft%d.h5' % epoch
    vgg.model.save_weights(results_path+latest_weights_filename)
    
print('Comleted %s fit operations' % n_epochs)


# # Generate Predictions
#Make predictions on the test dataset using our new model
batches, preds = vgg.test(test_path, batch_size=batch_size*2)

#vgg.test() generates two probabilities per image
#Need to check which column is cats and which is dogs
filenames = batches.filenames
print(preds[:5])
print(filenames[:5])

#Verify column ordering by viewing some images
from PIL import Image
Image.open(test_path + filenames[2])

# From the above, it appears that column 1 is cats and column 2 is dogs

#Save test results arrays for future use
save_array(results_path+'test_preds.dat', preds)
save_array(results_path+'filenames.dat', filenames)

# # Validate Predictions

#Calculate predictions on validation set for debugging of model
vgg.model.load_weights(results_path+latest_weights_filename)
val_batches, probs = vgg.test(valid_path, batch_size=batch_size)

#Find correct and incorrect examples
filenames = val_batches.filenames
expected_labels = val_batches.classes #Assigns '0' or '1'

#Round predictions to '0' or '1' to generate labels
our_predictions = probs[:,0]
our_labels = np.round(1 - our_predictions)


# Looking at examples of each of:
# 
# 1. A few correct labels at random
# 2. A few incorrect labels at random
# 3. Those with highest probability of being correct labels
# 4. Those with highest probability of being incorrect labels
# 5. Those with probability closest to 0.5, i.e. most uncertain labels

#Set-up for analysing above examples (not included in this file)
from keras.preprocessing import image

#Print out confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(expected_labels, our_labels)
plot_confusion_matrix(cm, val_batches.class_indices)


# # Submit Predictions to Kaggle
preds = load_array(results_path+'test_preds.dat')
filenames = load_array(results_path+'filenames.dat')

#Grab the dog prediction column (column 2)
isdog = preds[:,1]

#To play it safe, round down our edge predictions to avoid punishment from log-loss
isdog = isdog.clip(min=0.05, max=0.95)

#Extract imageIds from the filenames in our test directory
filenames = batches.filenames
ids = np.array([int(f[5:f.find('.')]) for f in filenames])
#Join the two columns into an array [imageId, isDog]
subm = np.stack([ids, isdog], axis=1)
subm[:5]

submission_file_name = 'submission1.csv'
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')
