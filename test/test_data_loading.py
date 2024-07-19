from utils.utils_data import load_data

def test_load_data():
    data_train_x, data_train_y, data_test_x, data_test_y, class_names = load_data()
    assert data_train_x is not None
    assert data_train_y is not None
    assert data_test_x is not None
    assert data_test_y is not None
    assert len(class_names) == 10

if __name__ == '__main__':
    test_load_data()