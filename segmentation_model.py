"""
Model for lung segmentation, which returns a 256x256 mask

Takes in a processed image
  - Dimensions of 256x256
  - Single Channel
  - Preferably normalised, I suppose

Returns
  - A mask with different values

"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

class Segmentation_Model:
  def __init__(self):
    self._model = keras.models.load_model('./models/segmentation_model.h5',custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef':dice_coef})

  @staticmethod
  def __read_img(file_path):
    _img = tf.io.read_file(file_path)
    _img = tf.image.decode_png(_img,channels=1)
    _img = tf.image.resize(_img,(256,256))
    _img = tf.expand_dims(_img,0)
    # _img = _img / 255
    return _img

  def get_mask(self, file_path, plot = False):
    img = self.__read_img(file_path)
    mask = self._model.predict(img)[0]
    
    if plot:
      plt.imshow(mask, cmap = 'gray')

    return mask
