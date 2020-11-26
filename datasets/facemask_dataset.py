from datasets.base_image_dataset import ImageDatasetBase
import pathlib
import tensorflow as tf

# the dataset class
class FacemaskDataset(ImageDatasetBase):
    def __init__(self, path, batch_size, img_height, img_width, augment, train_percentage, validation_percentage, test_percentage):
        super(FacemaskDataset, self).__init__(batch_size, img_height, img_width)

        # sets the path to the dataset
        self.path = pathlib.Path(path)

        # whether the images should be augmented
        self.augment = augment

        # creates the initial dataset from a directory containing all the frames
        # then mapped to the process_frame method to create the image from the path and give a label.
        self.data = tf.data.Dataset\
            .list_files(str(path/'*/*'), shuffle=True)\
            .map(self.process_frame, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # shuffles the dataset
        self.shuffle(256)

        # splits the data into train, validation, and test datasets.
        self.split_data_to_train_val_test(self.data, train_percentage, validation_percentage, test_percentage)

    def process_frame(self, img_filepath):
        # reads the image from disk as an encoded string.
        img = tf.io.read_file(img_filepath)
        # decodes it to an image.
        img = self.decode_img(img, channels=3, augment=self.augment)
        # checks the label for the image.
        masked = tf.strings.regex_full_match(img_filepath, ".*yes.*")
        return img, masked