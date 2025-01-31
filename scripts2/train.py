from data_handler import MNISTDataHandler

import network

# Initialize the data handler class
data_handler = MNISTDataHandler(val_size=0.2)

# Access data
x_train, y_train, x_val, y_val, x_test, y_test = data_handler.x_train, data_handler.y_train, data_handler.x_val, data_handler.y_val, data_handler.x_test, data_handler.y_test

print(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}, Test data shape: {x_test.shape}")


input_shape = x_train[0].shape
output_class_count = 10

model = network.build_network(input_shape, output_class_count)
model.summary()