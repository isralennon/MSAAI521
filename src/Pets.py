import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pylab as plt
from Globals import TENSORFLOW_DATASETS

class Pets:
    def __init__(self):
        self.data_dir = TENSORFLOW_DATASETS

    def load_data(self):
        self.dataset = tfds.load('oxford_iiit_pet', with_info=True, data_dir=self.data_dir)

    def read_and_preprocess(self, data):
        input_image = tf.image.convert_image_dtype(data['image'], tf.float32) # [0,1]
        input_image = tf.image.resize(input_image, (128, 128))
        input_mask = tf.image.resize(data['segmentation_mask'], (128, 128), method='nearest')
        input_mask -= 1 # {1,2,3} to {0,1,2}

        return input_image, input_mask

    def preprocess_data(self):
        self.train = self.dataset[0]['train'].map(self.read_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        self.test = self.dataset[0]['test'].map(self.read_and_preprocess)

    def inspect_data(self):
        for idx, (image, mask) in enumerate(self.train.take(3)):
            plt.subplot(2, 3, idx+1)
            plt.imshow(image)
            plt.title(f"Image {idx}")
            plt.axis('off')

            plt.subplot(2, 3, idx+4)
            plt.imshow(mask)
            plt.title(f"Mask {idx}")
            plt.axis('off')
        
        plt.show()


    def run(self):
        self.load_data()
        self.preprocess_data()
        self.inspect_data()