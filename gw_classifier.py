import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pycbc.frame import read_frame
from pycbc.filter import resample_to_delta_t, highpass
from pycbc.types.timeseries import TimeSeries
from create_template_database import create_templates
from pycbc_noise import create_noise


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

class_names = ["Detected", "Not Detected"]


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 8192, 1])

    conv1 = tf.layers.conv1d(
        inputs=input_layer,
        filters=16,
        kernel_size=16,
        activation=tf.nn.relu
    )
    # [8177, 16]

    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=4, strides=4)
    # [2044, 16]

    conv2 = tf.layers.conv1d(
        inputs=pool1,
        filters=32,
        kernel_size=8,
        activation=tf.nn.relu
    )
    # [2037, 32]

    pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=4, strides=4)
    # [509, 32]

    conv3 = tf.layers.conv1d(
        inputs=pool2,
        filters=64,
        kernel_size=8,
        activation=tf.nn.relu
    )
    # [502, 64]

    pool3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=4, strides=4)
    # [125, 64]

    pool3_flat = tf.reshape(pool3, [-1, 125 * 64])
    # [8000]

    dense = tf.layers.dense(inputs=pool3_flat, units=64, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=dense, units=2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    tf.summary.scalar('accuracy', accuracy[1])
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        logging_hook = tf.train.LoggingTensorHook({"accuracy": accuracy[1]}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == "__main__":

    print("Start")
    training_data = np.load("data/training_data.dat")
    training_data_noise = np.load("data/training_data_noise.dat")
    testing_data = np.load("data/testing_data.dat")
    testing_data_noise = np.load("data/testing_data_noise.dat")
    print("Loaded")

    training_features = []
    training_labels = []
    testing_features = []
    testing_labels = []

    for i in training_data:
        training_labels.append(i[1])
        training_features.append(i[0])
    for i in training_data_noise:
        training_labels.append(i[1])
        training_features.append(i[0])
    for i in testing_data:
        testing_labels.append(i[1])
        testing_features.append(i[0])
    for i in testing_data_noise:
        testing_labels.append(i[1])
        testing_features.append(i[0])

    print("Converting into numpy arrays")
    training_features = np.array(training_features)
    training_labels = np.array(training_labels)
    testing_features = np.array(testing_features)
    testing_labels = np.array(testing_labels)
    print("Preprocessing complete")

    batch_size = 10
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": training_features},
        y=training_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": testing_features},
        y=testing_labels,
        num_epochs=1,
        shuffle=False
    )

    gw_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                           model_dir="tmp/gw_cnn_model"
                                           )

    epochs = 10
    steps = int(epochs * len(training_features) / batch_size)
    print("Steps: {}".format(steps))
    while True:
        print("\n1. Train \n2. Evaluate \n3. Predict with wave \n")
        ch = int(input("Enter your choice:"))
        if ch == 1:
            gw_classifier.train(train_input_fn, steps=steps)
        if ch == 2:
            eval_results = gw_classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)
        if ch == 3:
            print("Enter the two masses:")
            m1 = float(input())
            m2 = float(input())
            nm = int(input("Enter noise multiplier:"))
            data = create_templates(m1, m2, noise_multiplier=nm)
            plt.plot(data)
            plt.show()
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.array([data])},
                num_epochs=1,
                shuffle=False
            )
            predictions = [p for p in gw_classifier.predict(predict_input_fn)]
            print(class_names[predictions[0]["classes"]])
            print(predictions[0])
