import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.data import Dataset
import math
from sklearn import metrics
from matplotlib import cm
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.options.display.max_rows = 10
pd.options.display.max_columns = 10
pd.options.display.float_format = '{:.1f}'.format
tf.logging.set_verbosity(tf.logging.ERROR)


def preprocess_features(dataframe):
    selected_features = dataframe[["latitude", "longitude", "housing_median_age",
                                  "total_rooms", "total_bedrooms", "population",
                                  "households", "median_income"]]
    processed_features = selected_features.copy()
    processed_features["rooms_per_person"] = dataframe["total_rooms"] / dataframe["population"]
    return processed_features


def preprocess_targets(dataframe):
    processed_targets = pd.DataFrame()
    processed_targets["median_house_value"] = dataframe["median_house_value"]/1000
    return processed_targets


def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(feature) for feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_model(learning_rate, steps, batch_size,
                training_examples, training_targets, validation_examples, validation_targets):
    periods = 10
    steps_per_period = steps/periods

    training_input_fn = lambda: my_input_fn(training_examples, training_targets, batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, shuffle=False, num_epochs=1)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, shuffle=False, num_epochs=1)

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    training_rmses = []
    validation_rmses = []
    validation_predictions = None
    training_predictions = None

    print("Training the model...")
    print("RMSE at each period:")
    for period in range(periods):
        linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
        training_rmse = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
        validation_rmse = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))
        print("\tPeriod {}: {:.2f}".format(period, training_rmse))
        training_rmses.append(training_rmse)
        validation_rmses.append(validation_rmse)

    print("Training finished")

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 1, 1)
    plt.title("Root Mean Square Errors")
    plt.xlabel("Period")
    plt.ylabel("RMSE")
    plt.plot(training_rmses, label="training")
    plt.plot(validation_rmses, label="validation")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Training scatter")
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.scatter(training_predictions, training_targets)

    plt.subplot(2, 2, 4)
    plt.title("Validation scatter")
    plt.xlabel("Predictions")
    plt.ylabel("Targets")
    plt.scatter(validation_predictions, validation_targets)

    plt.show()

    return linear_regressor


if __name__ == "__main__":
    california_housing_dataframe = pd.read_csv("data/california_housing_train.csv", sep=',')
    # print(california_housing_dataframe)
    # print(california_housing_dataframe.describe())
    california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

    training_examples = preprocess_features(california_housing_dataframe.head(1200))
    training_targets = preprocess_targets(california_housing_dataframe.head(1200))
    validation_examples = preprocess_features(california_housing_dataframe.tail(500))
    validation_targets = preprocess_targets(california_housing_dataframe.tail(500))

    train_model(learning_rate=0.00001, steps=600, batch_size=1,
                training_examples=training_examples, training_targets=training_targets,
                validation_examples=validation_examples, validation_targets=validation_targets)

# 0.00005, 500, 100 - total_rooms
# 0.03, 650, 100 - rooms_per_person
