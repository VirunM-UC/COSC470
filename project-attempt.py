import os
# suppress silly log messages from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow import keras

#MODEL

def load_data(data_folder, file_name):
    """Takes a pickled dataframe and returns a Tensorflow Dataset
    """
    df = pd.read_pickle(data_folder + file_name)
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    return ds


def get_n_records(batch):
    """returns number of records of batch"""
    return batch["City_name"].shape[0]
    
def get_images_labels(batch):
    """Takes Dataset batch and returns images, output_labels"""
    pass


def loss(logits, labels):
  """
	Calculates the cross-entropy loss after one forward pass.
	:param logits: during training, a matrix of shape (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	Softmax is applied in this function.
	:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
	:return: the loss of the model as a Tensor of shape (batch_size, loss)
	"""
  losses = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
  return losses

def accuracy(logits, labels):
	"""
	Calculates the model's prediction accuracy by comparing
	logits to correct labels – no need to modify this.
	:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

	:return: the accuracy of the model as a Tensor
	"""
	correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))



def train(model, train_dataset, validate_dataset=None):
    '''
    Trains the model
    '''

    train_acc = 0.

    def print_accuracy():
        if validate_dataset is not None:
            validate_acc = 0.

            validate_batches = 0.
            for batch in validate_dataset.batch(1024):
                validate_images, validate_labels = get_images_labels(batch)
                validate_logits = model.call(validate_images)

                validate_acc += accuracy(validate_logits, validate_labels)
                validate_batches += 1
            validate_acc /= validate_batches
        else:
            validate_acc = float('NaN')
        print(
            f'train accuracy {train_acc:.3f} validate accuracy {validate_acc:.3f}')

    for batch in train_dataset:
        images, labels = get_images_labels(batch)
        with tf.GradientTape() as tape:
            tape.watch(images)
            logits = model(images)

            l = loss(logits, labels)
            batch_loss = tf.reduce_sum(l)
            gradients = tape.gradient(batch_loss, model.trainable_weights)
            train_acc = accuracy(logits, labels)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))



def test(model, test_records):
    """
	Tests the model on the test inputs and labels.
	:param test_inputs: test data (all images to be tested),
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	"""
    for batch in test_records.batch(1024):
        test_images, test_labels = get_images_labels(batch)
        acc = accuracy(model.call(test_images), test_labels)
        print(f'test accuracy {acc:.3f}')


def main(data_folder):
    training_records = load_data(data_folder, 'training.tfr')
    validate_records = load_data(data_folder, 'validation.tfr')
    test_records = load_data(data_folder, 'testing.tfr')

    model = 0 #TODO
    model.optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.batch_size = 256 
    epochs = 5 
    # Iterate over epochs.
    for epoch in range(epochs):
        epoch_training_records = training_records.shuffle(buffer_size=256).batch(model.batch_size, drop_remainder=False)
        print("Start of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train(model, epoch_training_records, validate_records)

        test(model, test_records)

    test(model, test_records)

    model.save(data_folder + '/model.keras')

if __name__ == '__main__':
    data_folder = './data/'

    main(data_folder)

