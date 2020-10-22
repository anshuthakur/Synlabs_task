import os
from img_classifier import CustomImageClassifier
from tensorflow import keras
from config import SAVEPATH, DATAPATH


if __name__ == "__main__":
    
    # # Training a model and testing it. Uncomment the next three lines.
    # cls = CustomImageClassifier(DATAPATH)
    # model = cls.train(save=True)
    # cls.test()
    
    
    # # Loading a pre-trained model for testing
    # cls = CustomImageClassifier(DATAPATH)
    # model = keras.models.load_model(SAVEPATH)
    # cls.test(model)
    pass