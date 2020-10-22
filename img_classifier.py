import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import time
from config import SAVEPATH
from random import choice

class CustomImageClassifier:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.num_classes = len(os.listdir(data_path))
        self.names = os.listdir(data_path)
        
    
    def total_num_files(self):
        files = glob.glob(os.path.join(self.data_path, '*', "*.jpg"))
        print(f"Total number of files {len(files)}")
    
    def get_img_data_list(self):
        img_data_list=[]
        data_path = self.data_path
        data_dir_list = os.listdir(data_path)
        for dataset in data_dir_list:
            img_list=os.listdir(os.path.join(data_path, dataset))
            for img in img_list:
                if img.endswith("jpg"):
                    img_path = os.path.join(data_path ,dataset ,img)
                    img = load_img(img_path, target_size=(224, 224))
                    x = img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    img_data_list.append(x)
        return img_data_list
    
    def get_dataset(self):
        img_data_list= self.get_img_data_list()
        img_data = np.array(img_data_list)
        img_data = np.rollaxis(img_data,1,0)
        img_data = img_data[0]
        
        self.num_classes = len(os.listdir(self.data_path))
        num_of_samples = img_data.shape[0]
        labels = np.ones((num_of_samples,),dtype='int64')
        self.labels = labels
        names = os.listdir(self.data_path)
        # convert class labels to on-hot encoding
        Y = tf.keras.utils.to_categorical(labels, self.num_classes)
        x,y = shuffle(img_data,Y, random_state=2)
        
        return x,y
    
    def train(self, save=True):
        """
        The train function trains a classifier using a custom resnet model and returns the trained model 
        :params
        save - boolean - If true, the trained model is saved.
        returns
        keras model
        """
        self.total_num_files()
        x,y = self.get_dataset()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
        image_input = tf.keras.Input(shape=(224, 224, 3))
        model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
        print("Model Used:")
        print(model.name)
        last_layer = model.get_layer('avg_pool').output
        x= tf.keras.layers.Flatten(name='flatten')(last_layer)
        out = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='output_layer')(x)
        custom_resnet_model = tf.keras.Model(inputs=image_input,outputs= out)
        print("Customized resnet", custom_resnet_model.name)
        
        
        for layer in custom_resnet_model.layers[:-1]:
            layer.trainable = False
        
        custom_resnet_model.layers[-1].trainable

        custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        t=time.time()
        hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
        print('Training time: %s' % (t - time.time()))
        (loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

        print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
        self.model = custom_resnet_model
        
        if save:
            custom_resnet_model.save(SAVEPATH)
        
        y_pred = custom_resnet_model.predict(X_test)
        res = [np.argmax(a) == np.argmax(b) for a,b in zip(y_test, y_pred)]
        print("Accuracy - {:.2f}".format(sum(res) / len(res)))
        
        return custom_resnet_model
    
    def display_images(self, path):
        label = path.split("\\")[-2]
        print(f"Label - {label}")

        im = cv2.imread(path)
        im_new = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.imshow(im_new)
        plt.show()
        

    def test(self, model = None, image_path = None):
        """
        The test function takes an image from test file and predicts what class it belongs to
        :params
        model - the trained classifier model. If model has already been trained and no models have been passed as argumentit will take the pre-defined model for classification.
        image_path - the path of the image that has to be predicted. If no path is specified, it will randomly take an input from the dataset.
        :returns
        Nothing
        The function will first display the image, and then print the actual and predicted labels before exiting.
        """
        if not model:
            try:
                model = self.model
            except Exception as e:
                return "No model found. Please train a model before predicting, or send an existing model as argument."
                
        files = glob.glob(os.path.join(self.data_path, '*', "*.jpg"))
        if not image_path:
            test_file = choice(files)
        else:
            test_file = image_path
        self.display_images(test_file)
        
        img = load_img(test_file, target_size=(224, 224))
        img_data_list=[]
          
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_data_list.append(x)
        
        img_data = np.array(img_data_list)
        img_data = np.rollaxis(img_data,1,0)
        img_data = img_data[0]
        
        label = test_file.split("\\")[-2]
        
        
        ypred = model.predict(img_data)
        idx = np.argmax(ypred[0])
        label_pred = self.names[idx]
        
        print(f"Actual label: {label}\nPredicted label: {label_pred}")
        