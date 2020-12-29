from datasets.base_image_dataset import ImageDatasetBase
import pathlib
import tensorflow as tf

class FacemaskNetDataset(ImageDatasetBase):
    def __init__(self, path, batch_size, train_percentage, validation_percentage, test_percentage):
        super(FacemaskNetDataset, self).__init__(batch_size, 128, 128)
        self.batch_size = batch_size

        # sets the path to the dataset
        self.path = pathlib.Path(path)

        # creates the initial dataset from a directory containing all the images
        # then mapped to the process_image method to get X, Y
        self.data = tf.data.Dataset\
            .list_files(str(self.path/'**/*'), shuffle=True)\
            .map(self.process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffles the dataset
        self.shuffle(256)

        # splits the data into train, validation, and test datasets.
        self.split_data_to_train_val_test(self.data, train_percentage, validation_percentage, test_percentage)

    def process_image(self, image_path):
        # reads the image from disk as an encoded string
        img = tf.io.read_file(image_path)
        # decodes it to an image.
        img = self.decode_img(img, channels=3)

        def if_true(img, image_path):
            return img, [tf.constant(1, dtype=tf.float32), tf.constant(1, dtype=tf.float32), tf.constant(1, dtype=tf.float32)]

        def if_false(img, image_path):
            chin = tf.cond(tf.strings.regex_full_match(image_path, ".*Chin.*"), lambda: tf.constant(1, dtype=tf.float32), lambda: tf.constant(0, dtype=tf.float32))
            nose = tf.cond(tf.strings.regex_full_match(image_path, ".*Nose.*"), lambda: tf.constant(1, dtype=tf.float32), lambda: tf.constant(0, dtype=tf.float32))
            mouth = tf.cond(tf.strings.regex_full_match(image_path, ".*Mouth.*"), lambda: tf.constant(1, dtype=tf.float32), lambda: tf.constant(0, dtype=tf.float32))
            return img, [chin, mouth, nose]
        
        return img, tf.cond(tf.strings.regex_full_match(image_path, ".*Mask.jpg.*"), lambda: tf.constant(1, dtype=tf.float32), lambda: tf.constant(0, dtype=tf.float32))
        # return tf.cond(tf.strings.regex_full_match(image_path, ".*Mask.jpg.*"), lambda: if_true(img, image_path), lambda: if_false(img, image_path))