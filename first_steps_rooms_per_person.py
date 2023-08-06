import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.data import Dataset
import math
from sklearn import metrics
from matplotlib import pyplot as plt
from matplotlib import cm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
pd.options.display.float_format = '{:.1f}'.format
tf.logging.set_verbosity(tf.logging.ERROR)

california_housing_dataframe = pd.read_csv("data/california_housing_train.csv", sep=',')
# print(california_housing_dataframe)
# print(california_housing_dataframe.describe())
california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
skip_highest_value = california_housing_dataframe["median_house_value"] == 500001.0
skip_highest_value = pd.Series([not value for value in skip_highest_value])
california_housing_dataframe = california_housing_dataframe[skip_highest_value]
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe["rooms_per_person"] = california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"]
california_housing_dataframe["rooms_per_person"] = (california_housing_dataframe["rooms_per_person"]).apply(lambda x: min(x, 5))


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size, input_feature="rooms_per_person"):
    periods = 10
    steps_per_period = steps/10

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = "median_house_value"
    targets = california_housing_dataframe[my_label]
    features_columns = [tf.feature_column.numeric_column(my_feature)]

    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size=batch_size)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, shuffle=False, num_epochs=1)

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=features_columns,
        optimizer=my_optimizer,
        model_dir='tmp/model'
    )

    # writer = tf.summary.FileWriter('Summary')
    # writer.add_graph(tf.get_default_graph())
    # writer.flush()
    sample = california_housing_dataframe.sample(300)
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 2, 1)
    plt.title("Learned line")
    plt.xlabel(my_feature)
    plt.ylabel(my_label)
    plt.scatter(sample[my_feature], sample[my_label])
    colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

    root_mean_squared_errors = []
    predictions = None

    print("Training the model...")
    print("RMSE at each period:")
    for period in range(periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        root_mean_squared_error = math.sqrt(metrics.mean_squared_error(predictions, targets))
        print("\tPeriod {}: {:.2f}".format(period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)

        y_extents = np.array([0, sample[my_label].max()])

        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

        x_extents = (y_extents - bias) / weight
        x_extents = np.maximum(np.minimum(x_extents,
                                          sample[my_feature].max()),
                               sample[my_feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period])
    print("Training finished")

    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    calibration_data.describe()

    plt.subplot(2, 2, 2)
    plt.title("Root Mean Square Errors ({})".format(root_mean_squared_errors[-1]))
    plt.xlabel("Period")
    plt.ylabel("RMSE")
    plt.plot(root_mean_squared_errors)

    plt.subplot(2, 2, 3)
    plt.title("Predictions vs target scatter")
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.scatter(predictions, targets)

    plt.subplot(2, 2, 4)
    ax = plt.subplot(2, 2, 4)
    # plt.title("rooms_per_person histogram")
    # my_feature_data.hist(ax=ax, bins=int(my_feature_data.max()))
    california_housing_dataframe[["median_house_value"]].hist(ax=ax, bins=501)

    plt.show()


if __name__ == "__main__":
    train_model(learning_rate=0.03, steps=650, batch_size=100)

