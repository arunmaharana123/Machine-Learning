import numpy as np

train_data_file_name = "train.csv"
epoch_size = 100
theta0, theta1 = 0, 0
bias = 0;
dataset_size = 700
learning_rate = 0.05

# Load data set
data = np.loadtxt(train_data_file_name, delimiter=",")
# print(data)

x_axis_value = data[0:, 0:1]
# print(x_axis_value)

y_axis_value = data[0:, 1:]
# print(y_axis_value)

def gradientDescent(theta, bias):
    new_theta = theta - (learning_rate * (target - y_axis_value))
    new_bias = bias - (learning_rate * (target - y_axis_value))
    return new_theta, new_bias

for loop in range(0, epoch_size):
    target = (theta1 * x_axis_value) + bias
    loss = (1/(2*dataset_size))*np.sum((target - y_axis_value)**2)
    theta1, bias = gradientDescent(theta1, bias)
    # print(loop, " = ", bias)

print(bias)