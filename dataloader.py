"""Dataset loader as inspired by monodepth (https://github.com/mrharicot/monodepth)"""

import tensorflow as tf
import scipy

def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])

class OsvosDataloader(object):
    """Osvos dataloader"""

    def __init__(self, data_path, filenames_file, params, mode):
        """ data_path is the base_dir of the dataset.
filenames_file can be either a filename containing pair of space-separated relative image+segmentation paths like:
/JPEGImages/480p/car-shadow/00000.jpg /Annotations/480p/car-shadow/00000.png
or a list of such strings."""

        self.data_path = data_path
        self.params = params
        self.mode = mode

        self.image_batch = None
        self.segmentation_batch = None

        self.h, self.w = params.height, params.width

        if mode == 'test':
            limit_epochs = 1
        else:
            limit_epochs = None

        if isinstance(filenames_file, str):
            input_queue = tf.train.string_input_producer([filenames_file], shuffle=False, num_epochs=limit_epochs)
            line_reader = tf.TextLineReader()
            _, line = line_reader.read(input_queue)

            split_line = tf.string_split([line]).values

            with open(filenames_file, 'r') as fin:
                py_line = fin.readline()
                
        else:
            input_queue = tf.train.string_input_producer(filenames_file, shuffle=False, num_epochs=limit_epochs) 
            line = input_queue.dequeue()
            split_line = tf.string_split([line]).values

            py_line = filenames_file[0]

        # get the first image size
        if (self.h is None) or (self.w is None):
            py_split = py_line.split()
            py_img = scipy.misc.imread(self.data_path + py_split[0])
            self.h, self.w, _ = py_img.shape
            print('going to resize everything to {}x{}'.format(self.h, self.w))

        if mode == 'test':
            image_path = tf.string_join([self.data_path, split_line[0]])
            image_o = self.resize(self.read_image(image_path))
        else:
            image_path        = tf.string_join([self.data_path, split_line[0]])
            segmentation_path = tf.string_join([self.data_path, split_line[1]])

            image_o = self.resize(self.read_image(image_path))
            segmentation_o = self.resize(self.read_segmentation(segmentation_path))

        if mode == 'train':
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            image  = tf.cond(do_flip > 0.5,
                             lambda: tf.image.flip_left_right(image_o),
                             lambda: image_o)
            segmentation = tf.cond(do_flip > 0.5,
                                   lambda: tf.image.flip_left_right(segmentation_o),
                                   lambda: segmentation_o)

            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            image, segmentation = tf.cond(do_augment > 0.5,
                                          lambda: self.augment_image_pair(image, segmentation),
                                          lambda: (image, segmentation))

            image.set_shape( [None, None, 3])
            segmentation.set_shape([None, None, 1])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 0
            capacity = min_after_dequeue + 4 * params.batch_size
            self.image_batch, self.segmentation_batch = tf.train.shuffle_batch([image, segmentation],
                    params.batch_size, capacity, min_after_dequeue, params.num_threads,
                                                                allow_smaller_final_batch=False)

        elif mode == 'test':
            # self.image_batch = image_o
            self.image_batch = tf.expand_dims(image_o, 0)
            self.image_batch.set_shape([1, None, None, 3])
            self.image_path = image_path

        # self.image_batch_shape = tf.shape(self.image_batch)

    def augment_image_pair(self, image, segmentation):
        return image, segmentation

    def resize(self, image):
        image  = tf.image.resize_images(image,  [self.h, self.w], tf.image.ResizeMethod.AREA)
        return image
        
    def read_image(self, image_path):
        image  = tf.image.decode_jpeg(tf.read_file(image_path))
        image  = tf.image.convert_image_dtype(image,  tf.float32)
        return image

    def read_segmentation(self, image_path):
        image  = tf.image.decode_png(tf.read_file(image_path), channels=1)
        image  = tf.image.convert_image_dtype(image,  tf.float32)
        return image

