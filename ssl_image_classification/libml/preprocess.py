import os

import tensorflow as tf
import tensorflow_datasets as tfds


def download_dataset(dataset_name):
    """
    Downloads and splits tfds dataset into train and test

    Args:
        dataset_name

    Returns
        Train and test split of tfds dataset
    """
    dataset = tfds.load(name=dataset_name.lower())
    train = dataset["train"]
    test = dataset["test"]

    return train, test


def _list_to_tf_dataset(dataset, shape):
    def _dataset_gen():
        for sample in dataset:
            yield sample

    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={
            "image": tf.uint8,
            "label": tf.int64
            },
        output_shapes={
            "image": shape,
            "label": ()
            }
        )
    

def split_dataset(dataset, num_labeled_data, num_validations, num_classes):
    dataset = dataset.shuffle(buffer_size=10000)
    counter = [0 for _ in range(num_classes)]

    labeled, unlabeled, validation = [], [], []
    for sample in iter(dataset):
        label = int(sample["label"])
        counter[label] += 1
        if counter[label] <= (num_labeled_data / num_classes):
            labeled.append({
                "image": sample["image"],
                "label": sample["label"]
                })
            continue
        elif counter[label] <= (num_validations / num_classes + num_labeled_data / num_classes):
            validation.append({
                "image": sample["image"],
                "label": sample["label"]
                })
        else:
            unlabeled.append({
                "image": sample["image"],
                "label": tf.convert_to_tensor(-1, dtype=tf.int64)
            })

    shape = labeled[0]["image"].shape
    labeled = _list_to_tf_dataset(labeled, shape)
    unlabeled = _list_to_tf_dataset(unlabeled, shape)
    validation = _list_to_tf_dataset(validation, shape)

    return labeled, unlabeled, validation


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label):
    image = tf.image.encode_png(image)
    feature = {
        "image": _bytes_feature(image),
        "label": _int64_feature(int(label))
    }
    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample.SerializeToString()


def tf_serialize_example(sample):
    tf_string = tf.py_function(
        serialize_example,
        (sample["image"], sample["label"]),
        tf.string
    )
    return tf.reshape(tf_string, ())


def export_tfrecord_dataset(dataset_path, dataset):
    serialized_dataset = dataset.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(dataset_path)
    writer.write(serialized_dataset)


def _parse_function(sample):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(sample, feature_description)


def load_tfrecord_dataset(dataset_path):
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def normalize_image(image, start=(0., 255.), end=(-1., 1.)):
    image = (image - start[0]) / (start[1] - start[0])
    image = image * (end[1] - end[0]) + end[0]
    return image


def process_parsed_dataset(dataset, num_classes):
    images, labels = [], []

    for sample in iter(dataset):
        decoded_image = tf.io.decode_png(sample["image"], channels=3, dtype=tf.uint8)
        normalized_image = normalize_image(tf.cast(decoded_image, dtype=tf.float32))
        images.append(normalized_image)
        one_hot_label = tf.one_hot(sample["label"], depth=num_classes, dtype=tf.float32)
        labels.append(one_hot_label)
    return tf.data.Dataset.from_tensor_slices({
        "image": images,
        "label": labels
        })


def fetch_dataset(args, log_dir):
    dataset_path = f"{log_dir}/datasets"

    # Create dataset path if it does not exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Creating Datasets
    if any([not os.path.exists(f"{dataset_path}/{split}.tfrecord") for split in ["trainX", "trainU", "validation", "test"]]):
        # download train and test data
        train, test = download_dataset(dataset_name=args["dataset"])

        # Split train data into labeld, unlabeld and validation data
        trainX, trainU, validation = split_dataset(
            train,
            args["num_lab_samples"],
            args["val_samples"],
            args["num_classes"]
            )

        # Saving trainX, trainU, validation and test data as .tfrecord files
        for name, dataset in [("trainX", trainX), ("trainU", trainU), ("validation", validation), ("test", test)]:
            export_tfrecord_dataset(f"{dataset_path}/{name}.tfrecord", dataset)


    # Loading datasets from .tfrecord files
    parsed_trainX = load_tfrecord_dataset(f"{dataset_path}/trainX.tfrecord")
    parsed_trainU = load_tfrecord_dataset(f"{dataset_path}/trainU.tfrecord")
    parsed_validation = load_tfrecord_dataset(f"{dataset_path}/validation.tfrecord")
    parsed_test = load_tfrecord_dataset(f"{dataset_path}/test.tfrecord")

    trainX = process_parsed_dataset(parsed_trainX, args["num_classes"])
    trainU = process_parsed_dataset(parsed_trainU, args["num_classes"])
    validation = process_parsed_dataset(parsed_validation, args["num_classes"])
    test = process_parsed_dataset(parsed_test, args["num_classes"])

    return trainX, trainU, validation, test