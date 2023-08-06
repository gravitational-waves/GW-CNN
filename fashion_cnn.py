import mnist_reader
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def plot_image(probabilities, true_class, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_class = np.argmax(probabilities)
    if predicted_class == true_class:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("True:{} {:2.0f}% ({})".format(class_names[true_class],
                                              100 * np.max(probabilities),
                                              class_names[predicted_class]),
               color=color)


def plot_value_array(probabilities, true_class):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    barplot = plt.bar(range(10), probabilities, color="#777777")
    plt.ylim([0, 1])
    predicted_class = np.argmax(probabilities)
    barplot[predicted_class].set_color('red')
    barplot[true_class].set_color('blue')
    plt.xticks(range(10), class_names, rotation=45)


def analyze_image(i):
    single_predict_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array([test_images[num]])},
        y=np.array([test_labels[num]]),
        num_epochs=1,
        shuffle=False)
    prediction = list(mnist_classifier.predict(input_fn=single_predict_fn))[0]
    true_label = prediction["classes"]
    probabilities = prediction["probabilities"]
    img = test_images[i].reshape(28, 28)

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plot_image(probabilities, true_label, img)
    plt.subplot(1, 2, 2)
    plot_value_array(probabilities, true_label)
    plt.show()
    # img = test_images[i]
    # img = (np.expand_dims(img, 0))
    # plot_value_array(0, predictions, test_labels)
    # _ = plt.xticks(range(10), class_names, rotation=45)
    # plt.show()


if __name__ == "__main__":
    train_data = mnist_reader.load_mnist("data/fashion", kind="train")
    test_data = mnist_reader.load_mnist("data/fashion", kind="t10k")

    print(train_data[1])
    train_images = train_data[0]/np.float32(255)
    train_labels = train_data[1].astype(np.int32)
    test_images = test_data[0]/np.float32(255)
    test_labels = test_data[1].astype(np.int32)

    print(test_images.shape)
    print(np.array([test_images[0]]).shape)
    print(test_labels.shape)
    print(np.array([test_labels[0]]).shape)
    exit()

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                              model_dir="tmp/mnist_convnet_model")
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_images},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_images},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    mnist_classifier.train(train_input_fn, steps=1300)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    while True:
        num = int(input("Enter image number:"))
        analyze_image(num)
