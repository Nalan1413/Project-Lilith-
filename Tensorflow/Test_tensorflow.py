# No Truce With The Furies
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Getting the dataset
data = pd.read_csv('/Users/nalansuo/Downloads/archive/EN_Dataset/en_lpor_classification.csv')

pro_data = pd.DataFrame()
pro_data["Good_Family_Relationship"] = data["Good_Family_Relationship"]
pro_data["Time_with_Friends"] = data["Time_with_Friends"]

# Adding two columns together
pro_data["Alcohol_day_per_week"] = data["Alcohol_Weekdays"] + data["Alcohol_Weekends"]
pro_data["sum_grade"] = data["Grade_1st_Semester"] + data["Grade_2nd_Semester"]

# Check out the data
# print(pro_data)

# Make a scatter graph and checking it
plt.hexbin(pro_data["Alcohol_day_per_week"], pro_data["sum_grade"], gridsize=50, cmap='viridis', bins='log')
plt.colorbar(label='log10(counts)')
plt.xlabel('Alcohol_day_per_week')
plt.ylabel('sum_grade')
plt.title('Scatter Plot with Heatmap')
# plt.show()

# Setting x and y for inputs and outputs
x = pro_data.iloc[:, :-1]
y = pro_data.sum_grade

# Tensorflow modeling using Sequential model
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation="relu"),
                             tf.keras.layers.Dense(1)])
# Show model
# model.summary()

# Using adam to optimize, loss function Mean squared error (mse)
model.compile(optimizer="adam", loss="mse")

# Training with fit 1000 times
history = model.fit(x, y, epochs=1000)

# Using model.predict to predict
input_data = pd.DataFrame({'Good_Family_Relationship': [5], 'Time_with_Friends': [5], 'Alcohol_day_per_week': [0]})
prediction = model.predict(input_data)

# Print out the prediction
print("The grade prediction is:", prediction)
