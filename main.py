from utils.utils_DL import train_model, evaluate_model
from utils.utils_data import load_data, inspect_data, preprocess_data

def main():
    data_train_x, data_train_y, data_test_x, data_test_y, class_names=load_data()
    
    # Inspect the data
    inspect_data(data_train_x, data_train_y, class_names)

    # Preprocess the data
    train_x, val_x, test_x, train_y, val_y, test_y = preprocess_data((data_train_x, data_train_y, data_test_x, data_test_y))

    # Train the model
    model = train_model(train_x, train_y, val_x, val_y)

    # Evaluate the model
    evaluate_model(model, test_x, test_y, class_names)

if __name__ == '__main__':
    main()

# Example of use
# python main.py --epochs=10 --batch_size=250