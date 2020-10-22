import os
from img_classifier import CustomImageClassifier
from tensorflow import keras

if __name__ == "__main__":
    data_path = os.path.join(os.getcwd(), "data")
    cls = CustomImageClassifier(data_path)
    #model = cls.train()
    model_path =os.path.join(os.getcwd(), "saved_models", "classification.h5") 
    #model.save(model_path)
    model = keras.models.load_model(model_path)
    
    cls.test(model)