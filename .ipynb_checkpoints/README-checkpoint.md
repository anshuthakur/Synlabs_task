# Synlabs_task
 Assignment Task
 
### How to run the code?
 - To run the code, download the file img_classifier.py, and the data folder (in case you don't have an existing folder of images already) and run the following lines in Jupyter notebook or in a separate python file.
 
```
from img_classifier import CustomImageClassifier

cls = CustomImageClassifier(/path/to/image/dataset)
model = cls.train()
cls.test()
```
 - After training the model gets saved in saved_models folder, you can use the trained model for predicting your image classes.

```
from img_classifier import CustomImageClassifier
from tensorflow import keras

cls = CustomImageClassifier(/path/to/image/dataset)
model = keras.models.load_model(/saved/model/path)

cls.test(model)
```