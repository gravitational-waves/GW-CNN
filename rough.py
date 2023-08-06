import numpy as np
import pandas as pd

pd.options.display.max_rows = 10
pd.options.display.max_columns = 15
pd.options.display.float_format = '{:.1f}'.format

# ------ Trying out extents ----------------
# y_extents = np.array([0, 500])
# weight = 3
# bias = 10
# x_extents = (y_extents - bias)/weight
# print(x_extents)
#
# feature_max = 150
# feature_min = 2
#
# min = np.minimum(x_extents, feature_max)
# print(min)
# x_extents = np.maximum(min, feature_min)
# print(x_extents)
#
# y_extents = x_extents*weight + bias
# print(y_extents)
# -------------------------------------------

california_housing_dataframe = pd.read_csv("data/california_housing_train.csv", sep=',')
california_housing_dataframe["rooms_per_person"] = california_housing_dataframe["total_rooms"]/california_housing_dataframe["population"]
# print(california_housing_dataframe.describe())
# my_feature_data = california_housing_dataframe[["population", "total_rooms"]]
# print(my_feature_data)
skip_highest_value = california_housing_dataframe["median_house_value"] == 500001.0
skip_highest_value = pd.Series([not value for value in skip_highest_value])
m_dataframe = california_housing_dataframe[skip_highest_value]
print(m_dataframe.describe())
