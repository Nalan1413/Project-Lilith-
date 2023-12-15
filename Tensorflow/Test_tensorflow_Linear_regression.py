# No Truce With The Furies
# This is a code that predicts students grades in relation to their use of alcohol
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Getting the dataset
data = pd.read_csv('/Users/nalansuo/Downloads/archive/EN_Dataset/en_lpor_classification.csv')

# Triming data
pro_data = (data.drop(data.columns[:25], axis=1)
            .drop(data.columns[27:29], axis=1))  # Deleting useless colums using drop()

# Adding two columns together and deleting the original ones
pro_data["Alcohol_day_per_week"] = pro_data["Alcohol_Weekdays"] + pro_data["Alcohol_Weekends"]
pro_data = pro_data.drop(pro_data.columns[:2], axis=1)

# Same here
pro_data["sum_grade"] = pro_data["Grade_1st_Semester"] + pro_data["Grade_2nd_Semester"]
pro_data = pro_data.drop(pro_data.columns[:2], axis=1)

# Check out the data
# print(pro_data)

# Make a scatter graph and checking it
plt.scatter(pro_data.Alcohol_day_per_week, pro_data.sum_grade)
# plt.show()

# Setting x and y for inputs and outputs
x = pro_data.Alcohol_day_per_week
y = pro_data.sum_grade

# Tensorflow modeling using Sequential model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))  # Adding layers, Dense = "ax+b" (1 output, 1 input)

# Show model
# model.summary()

# Using adam to optimize, loss function Mean squared error (mse)
model.compile(optimizer="adam", loss="mse")

# Training with fit 2500 times
history = model.fit(x, y, epochs=2500)

# Using model.predict to predict
prediction = model.predict(pd.Series([7]))

# Print out the prediction
print("The grade prediction is:", prediction)
