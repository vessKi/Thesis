import os
import json
import tensorflow as tf

# Paths
data_dir = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset\quickdraw_dataset'
ndjson_dir = data_dir
tfrecord_dir = os.path.join(data_dir, 'tfrecords')
os.makedirs(tfrecord_dir, exist_ok=True)

# Function to create TFRecord from NDJSON
def create_tfrecords(ndjson_dir, tfrecord_dir):
    for category in os.listdir(ndjson_dir):
        if category.endswith('.ndjson'):
            category_name = category.split('.')[0]
            ndjson_file = os.path.join(ndjson_dir, category)
            tfrecord_file = os.path.join(tfrecord_dir, f'{category_name}.tfrecords')

            with tf.io.TFRecordWriter(tfrecord_file) as writer:
                with open(ndjson_file, 'r') as f:
                    for line in f:
                        example = json.loads(line)
                        inkarray = example['drawing']
                        stroke_lengths = [len(stroke[0]) for stroke in inkarray]
                        total_points = sum(stroke_lengths)
                        flattened_strokes = [(x, y, i) for i, (xs, ys) in enumerate(inkarray) for x, y in zip(xs, ys)]
                        
                        # Features
                        features = {
                            'ink': tf.train.Feature(float_list=tf.train.FloatList(value=[coord for point in flattened_strokes for coord in point])),
                            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[total_points, 3])),
                            'class_index': tf.train.Feature(int64_list=tf.train.Int64List(value=[category_index]))
                        }

                        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(example_proto.SerializeToString())

create_tfrecords(ndjson_dir, tfrecord_dir)
