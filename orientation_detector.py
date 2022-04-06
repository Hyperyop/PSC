from numpy import void
import keras,os
import numpy as np
import tensorflow as tf
class detector:
    model=0
    def __init__(self,path_to_model="VGG19.model"):
        try:
            self.model = tf.keras.models.load_model(path_to_model)
        except ImportError:
            print("Failed to import check if missing h5 file")
        except IOError:
            print("Invalid savefile")
        print("finished importing model")
    def __call__(self,image):
        if (len(image.shape)==4):
            predictions=self.model.predict(image)
            result = ["droite" if np.argmax(i)==0 else "gauche" for i in predictions]
            return result
        prediction = self.model(np.expand_dims(image,axis=0))
        if (np.argmax(prediction)==0):
            return "droite"
        else:
            return "gauche"
if __name__=="__main__":
    print("please use as a class")
