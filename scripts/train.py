import argparse
import json
import os
import keras
from util import load_data, preprocess_data, create_experiment_dir
from network import build_network

def train(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create experiment directory
    save_dir = create_experiment_dir(config['save_dir'], config['experiment_name'])

    # Load and preprocess data
    data = load_data(config['dataset'])
    train_x, val_x, test_x, train_y, val_y, test_y = preprocess_data(data)

    # Build model
    model = build_network(input_shape=(32, 32, 1), output_class_count=10)

    # Compile model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, 'model-{epoch:02d}.h5')),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)
    ]

    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(val_x, val_y),
        callbacks=callbacks
    )

    # Save final model and history
    model.save(os.path.join(save_dir, 'final_model.h5'))
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the configuration file (JSON)")
    args = parser.parse_args()

    train(args.config)
