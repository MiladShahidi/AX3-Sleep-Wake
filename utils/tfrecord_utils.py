import tensorflow as tf
import numpy as np
import pandas as pd
import os


def to_feature(value):
    """Returns a `feature` from a value. For internal use by other functions in this module"""
    def _bytes_feature(value):
        """Returns a `bytes_list` from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
        """Returns a `float_list` from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int_feature(value):
        """Returns an `int64_list` from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    if isinstance(value, list):
        sample_value = value[0]
    else:
        sample_value = value
        value = [value]

    # TODO: This type checking feels hacky. Are np numerical types the same on all machines and Operating Systems?
    # This used to be np.int64, but it didn't work with Pandas dataframes.
    if isinstance(sample_value, np.int) or isinstance(sample_value, np.int32) or isinstance(sample_value, np.int64):
        return _int_feature(value)
    # This used to be np.float32, but it didn't work with Pandas dataframes.
    elif isinstance(sample_value, np.float) or isinstance(sample_value, np.float32):
        return _float_feature(value)
    elif isinstance(sample_value, bytes):
        return _bytes_feature(value)
    elif isinstance(sample_value, str):
        value = [v.encode('utf-8') for v in value]
        return _bytes_feature(value)
    else:
        raise Exception("Encountered unsupported type {}".format(type(sample_value)))


def encode_tf_example(raw_features):
    """
    Returns a `tf.Example` object encapsulating `raw_features`. For internal use by other functions in this module.
    """
    features = {
        feature_key: to_feature(feature_value) for feature_key, feature_value in raw_features.items()
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def pandas_to_tf_seq_example_list(df, group_id_columns):
    """
    This function converts a pandas DataFrame into a list of Tensorflow `SequenceExample` objects.
    It performs a groupby using `group_id_column` and collects the values of all other columns into lists (similar to
    PySpark's `collect_list`). Then, features that become a list of lists (see `list_feature` in the example below),
    will be stored in the `feature_lists` component of the `tf.SequenceExample` object, while all other features
    (the ones that were originally scalar values and have now become a list of scalars) are stored in the
    `context` component. See https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample

    You only need to use `tf.SequenceExample` if you need list of lists to store your data. This happens when some
    columns in the dataset contain a list of values in each row (see `list_feature` column below). In all other cases,
    use `tf.Example` which is considerably easier to work with and results in less verbose code.

    Example:
    ```python

    df = pd.DataFrame({
        'id': [1, 1, 2, 2, 3],
        'int_feature': [10, 11, 21, 22, 31],
        'list_feature': [[100, 101], [110, 111, 112], [200, 201], [210, 211], [300]]
    })

    seq_ex_list = pandas_to_tf_seq_example_list(df, group_id_column='id')
    print(seq_ex_list[0])
    ```
    Expected Output:

        context {
          feature {
            key: "id"
            value {
              int64_list {
                value: 1
              }
            }
          }
          feature {
            key: "int_feature"
            value {
              int64_list {
                value: 10
                value: 11
              }
            }
          }
        }
        feature_lists {
          feature_list {
            key: "list_feature"
            value {
              feature {
                int64_list {
                  value: 100
                  value: 101
                }
              }
              feature {
                int64_list {
                  value: 110
                  value: 111
                  value: 112
                }
              }
            }
          }
        }

    The output corresponding to each group will be a SequenceExample object consisting of two components:
    1) context
    2) feature_lists

    Notice that the column that contained a list per row, is converted into a list of lists (list_feature here).

    The list of sequence examples can be passed to the `write_to_tfrecord` method.

    Args:
        df: A Pandas DataFrame.
        group_id_columns: A list of column names to be used for groupby.

    Returns:
        A list of `tf.train.SequenceExample` objects each element of which corresponds to a group in
            df.groupby(group_id_column)
    """
    assert isinstance(group_id_columns, list), "group_id_columns must be a list."

    def to_tf_seq_example(collected_row):
        row_dict = collected_row.to_dict()
        sequence_features_dict = {}

        # columns we group by must always appear as scalars, not lists
        context_features_dict = {col: row_dict[col][0] for col in group_id_columns}
        row_dict = {k: v for k, v in row_dict.items() if k not in group_id_columns}  # remove group_id_columns

        for feature_name, feature_value in row_dict.items():
            # Since we receive one row of the results of a groupby operation, each column is a list. That much
            # is certain. However, some columns are a 1-D list while other are nested 2-D lists.
            if isinstance(feature_value[0], list):  # Examine one value to see whether this is a list or a list of lists
                sequence_features_dict[feature_name] = feature_value
            else:
                context_features_dict[feature_name] = feature_value

        sequence_features_dict = {k: tf.train.FeatureList(feature=[to_feature(element) for element in v])
                                  for k, v in sequence_features_dict.items()}
        sequence_features_dict = tf.train.FeatureLists(feature_list=sequence_features_dict)

        context_features_dict = {k: to_feature(v) for k, v in context_features_dict.items()}
        seq_example = tf.train.SequenceExample(
            context=tf.train.Features(feature=context_features_dict),
            feature_lists=sequence_features_dict
        )

        return seq_example

    collected_df = pd.DataFrame()
    for col in df.columns:
        collected_df[col] = df.groupby(group_id_columns)[col].apply(list)

    # ToDo: This function decides how to split features between the "context" and the "sequence" components of the
    #  sequence example. It is probably better to either add this as an argument so that the caller can specify that,
    #  which would make the caller code more readable, or have this function return the resulting allocation like
    #  component_features = ['context_feature_1' , 'context_feature_2']
    #  sequence_features = ['seq_feature_1', 'seq_feature_2']
    return collected_df.apply(to_tf_seq_example, axis=1).to_list()


def write_to_tfrecord(data, path, filename_prefix, records_per_shard=10 ** 4):
    """
    Writes a list of `tf.Example's or `tf.SequenceExample`s to tfrecord file(s).

    Example:

    ```python
    df = pd.DataFrame({
        'id': [1, 1, 2, 2, 3],
        'int_feature': [10, 11, 21, 22, 31],
        'str_feature': ['1A', '1B', '2A', '2B', '3A']
    })

    ex_list = pandas_to_tf_example_list(df, group_id_column='id')

    write_to_tfrecord(data=ex_list, path='.', filename_prefix='temp')

    feature_spec = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'int_feature': tf.io.VarLenFeature(tf.int64),
        'str_feature': tf.io.VarLenFeature(tf.string)
    }

    dataset = tf.data.TFRecordDataset(['temp.tfrecord'])

    for serialized_string in dataset:
        example = tf.io.parse_example(serialized_string, feature_spec)
        # `VarLenFeatures` are parsed as `SparseTensor`s. The next step is not necessary, but makes output more readable
        for feature, tensor in example.items():
            if isinstance(tensor, tf.SparseTensor):
                example[feature] = tf.sparse.to_dense(tensor)
            print(feature, ': ', example[feature])
        print('*'*40)
    ```

    Expected Output:

        int_feature :  tf.Tensor([10 11], shape=(2,), dtype=int64)
        str_feature :  tf.Tensor([b'1A' b'1B'], shape=(2,), dtype=string)
        id :  tf.Tensor(1, shape=(), dtype=int64)
        ****************************************
        int_feature :  tf.Tensor([21 22], shape=(2,), dtype=int64)
        str_feature :  tf.Tensor([b'2A' b'2B'], shape=(2,), dtype=string)
        id :  tf.Tensor(2, shape=(), dtype=int64)
        ****************************************
        int_feature :  tf.Tensor([31], shape=(1,), dtype=int64)
        str_feature :  tf.Tensor([b'3A'], shape=(1,), dtype=string)
        id :  tf.Tensor(3, shape=(), dtype=int64)
        ****************************************

    Args:
        data: A list of `tf.Example` or `tf.SequenceExample` objects.
        filename_prefix: File name prefix. In case data is sharded, file numbers will be appended to this prefix.
        records_per_shard: Number of records to write in each file.

    Returns:
        None
    """
    shard_boundaries = [k * records_per_shard for k in range(len(data) // records_per_shard + 1)]
    if shard_boundaries[-1] < len(data):
        shard_boundaries.append(len(data))  # in case the number of records is not an exact multiple of shard size
    num_shards = len(shard_boundaries) - 1
    for i, (shard_start, shard_end) in enumerate(zip(shard_boundaries, shard_boundaries[1:])):
        os.makedirs(path, exist_ok=True)
        shard_numbering = f'_{i+1}_of_{num_shards}' if num_shards > 1 else ''
        filename = filename_prefix + shard_numbering + '.tfrecord'
        filename = os.path.join(path, filename)
        with tf.io.TFRecordWriter(filename) as writer:
            for record in data[shard_start:shard_end]:
                writer.write(record.SerializeToString())
