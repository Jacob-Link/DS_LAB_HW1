from gradientBoosting import x_y_test, calc_f1
import pickle

from feature_selection import top_ten_non_missing, keep_all
from feature_transformation import std_mean_transform, impute_mean




def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
        print(f'>>> successfully loaded {file_name}')
    return model


if __name__ == '__main__':
    # model_name = "gradient_boosting_071.pkl"
    model_name = "random_forest_0676.pkl"
    selection = keep_all
    transformation = std_mean_transform

    X_test, y_test = x_y_test(selection, transformation)
    model = load_model(model_name)

    predictions = model.predict(X_test)
    f1 = calc_f1(predictions, y_test)

