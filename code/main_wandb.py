import wandb
from wandb.integration.keras import WandbCallback
import tensorflow as tf
from tensorflow import keras
from absl import app, flags
from util import load_data, inspect_data, preprocess_data, build_lenet5, evaluate_model, print_training_summary, plot_history

# Define hyperparameters as flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 12, 'Number of epochs to train the model')
flags.DEFINE_integer('batch_size', 250, 'Batch size for training')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the optimizer')

# Modified train_model function
def train_model(train_x, train_y, val_x, val_y, epochs, batch_size, learning_rate):
    # Build and compile the model
    model = build_lenet5()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # WandbCallback with explicit parameters set
    wandb_callback = WandbCallback(
        save_model=False,   # Disabling local model save
        log_weights=True,   # Log weights to Wandb
        log_gradients=True, # Log gradients to Wandb
        save_graph=True     # Log graph structure to Wandb
    )
    
    # Train the model
    history = model.fit(
        train_x, 
        train_y, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(val_x, val_y), 
        verbose=1, 
        callbacks=[wandb_callback])
    
    # Plot of the training history
    plot_history(history)

    return model, history

def main(argv):
    wandb.login()
    # Initialize wandb
    wandb.init(
        project="MNIST-CNN",
        config={
            "epochs": FLAGS.epochs,
            "batch_size": FLAGS.batch_size,
            "architecture": "LeNet-5",
            "dataset": "MNIST",
            "optimizer": "Adam",
            "learning_rate": FLAGS.learning_rate
        },
        settings=wandb.Settings(_disable_stats=True),
    )
    
    # Load data
    data_train_x, data_train_y, data_test_x, data_test_y, class_names = load_data()
    
    # Inspect the data
    #inspect_data(data_train_x, data_train_y, class_names)

    # Preprocess the data
    train_x, val_x, test_x, train_y, val_y, test_y = preprocess_data((data_train_x, data_train_y, data_test_x, data_test_y))

    # Train the model
    model, history = train_model(train_x, train_y, val_x, val_y, FLAGS.epochs, FLAGS.batch_size,FLAGS.learning_rate) 

    # Evaluate the model
    evaluate_model(model, test_x, test_y, class_names)
    
    wandb.finish()

if __name__ == '__main__':
    app.run(main)

# Example of use
# python main_wandb.py --epochs=10 --batch_size=250 --learning_rate=0.001
