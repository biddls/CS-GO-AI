import tensorflow as tf
import glob, os
from time import sleep

pool_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
#use the CSV then in vott

os.chdir("dat saved/CS-GO-TFRecords-export")
files = glob.glob("*.tfrecord")

files = tf.data.TFRecordDataset(files)

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

for x in files:
    x = _bytes_feature(x)


print(type(files))

sleep(1)
for x in files:
    print(type(x))
    x = tf.train.Feature(bytes_list = x)
    break