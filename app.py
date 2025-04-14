from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import time

class SVC:
    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)
        svm_SVC = LinearSVC(random_state=42, max_iter=1000, class_weight='balanced')
        svm_SVC.fit(X_train, y_train)

        y_pred_SVC = svm_SVC.predict(X_test)
        accuracy_SVC = accuracy_score(y_test, y_pred_SVC)
        report = classification_report(y_test, y_pred_SVC, output_dict=True)

        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        cm = confusion_matrix(y_test, y_pred_SVC)
        tn, fp, fn, tp = (0, 0, 0, 0)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

        return svm_SVC, accuracy_SVC, precision, recall, f1_score, cm

    def evaluate_model_SVC(self, model, X_test, y_test):
        y_pred_SVC = model.predict(X_test)
        accuracy_SVC = accuracy_score(y_test, y_pred_SVC)
        report = classification_report(y_test, y_pred_SVC, output_dict=True)

        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        cm = confusion_matrix(y_test, y_pred_SVC)
        return accuracy_SVC, precision, recall, f1_score, cm

class FL:
    @staticmethod
    def federated_averaged_svm_SVC(models_SVC, client_data_sizes, classes_SVC):
        total_data = sum(client_data_sizes)
        weights = np.array([size / total_data for size in client_data_sizes])

        coef_shapes = [model.coef_.shape for model in models_SVC]
        if len(set(coef_shapes)) > 1:
            raise ValueError("Mismatch in coef_ shapes across models.")
        
        print("All models have the same coef_ shape.")

        print(f"Shapes of coef_ in all models: {coef_shapes}")

        avg_coef_SVC = np.average([model.coef_ for model in models_SVC], axis=0, weights=weights)

        avg_intercept_SVC = np.average([model.intercept_ for model in models_SVC], axis=0, weights=weights)

        avg_model_SVC = LinearSVC(random_state=42, max_iter=10000)
        avg_model_SVC.coef_ = avg_coef_SVC
        avg_model_SVC.intercept_ = avg_intercept_SVC
        avg_model_SVC.classes_ = classes_SVC

        return avg_model_SVC


    @staticmethod
    def federated_learning_SVC(models_SVC, client_data_sizes, X_test_SVC, y_test_SVC, classes_SVC):
        start_time = time.time()
        if len(models_SVC) < 2:
            return jsonify({
            "status": "error",
            "message": "Models not found",
            "code": 400
        }), 400
        
        global_model_SVC = FL.federated_averaged_svm_SVC(models_SVC, client_data_sizes, classes_SVC)
        
        accuracy_SVC, precision, recall, f1_score, confusion_mat = FL.evaluate_model_SVC(global_model_SVC, X_test_SVC, y_test_SVC)
        end_time = time.time()
        return {
            'Accuracy': accuracy_SVC,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Confusion Matrix': confusion_mat.tolist(),
            'Time': end_time-start_time,
        }

    @staticmethod
    def evaluate_model_SVC(model, X_test_SVC, y_test_SVC):
        y_pred_SVC = model.predict(X_test_SVC)
        accuracy_SVC = accuracy_score(y_test_SVC, y_pred_SVC)
        report = classification_report(y_test_SVC, y_pred_SVC, output_dict=True)

        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']

        cm = confusion_matrix(y_test_SVC, y_pred_SVC)
        return accuracy_SVC, precision, recall, f1_score, cm

    @staticmethod
    def predict(data):
        return np.max(data, axis=1)


svc1 = SVC()
fl = FL()

app = Flask(__name__)
models_SVC = []
data1 = pd.read_csv('apidf.csv')


def train_models():
    global models_SVC
    start_time = time.time()

    X = data1.iloc[:, :-1]
    y = data1.iloc[:, -1]

    model1, accuracy, precision, recall, f1_score, cm = svc1.train_model(X[:15000], y[:15000])
    models_SVC.append(model1)

    model2, accuracy, precision, recall, f1_score, cm = svc1.train_model(X[15000:30000], y[15000:30000])
    models_SVC.append(model2)

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")


train_models()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/model1', methods=['POST'])
def run_model1():
    try:
        start_time = time.time()

        X_model1 = data1.iloc[:15000, :-1]
        y_model1 = data1.iloc[:15000, -1]

        print(f"Model 1 Features Shape: {X_model1.shape}, Target Shape: {y_model1.shape}")

        model1, accuracy, precision, recall, f1_score, cm = svc1.train_model(X_model1, y_model1)
        models_SVC.append(model1)
        end_time = time.time()

        return jsonify({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Confusion Matrix': cm.tolist(),
            'Time': end_time-start_time,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/model2', methods=['POST'])
def run_model2():
    try:
        X_model2 = data1.iloc[15000:30000, :-1]
        y_model2 = data1.iloc[15000:30000, -1]

        print(f"Model 2 Features Shape: {X_model2.shape}, Target Shape: {y_model2.shape}")
        
        start_time = time.time()
        model2, accuracy, precision, recall, f1_score, cm = svc1.train_model(X_model2, y_model2)
        models_SVC.append(model2)

        end_time = time.time()
        return jsonify({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Confusion Matrix': cm.tolist(),
            'Time': end_time-start_time,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400



@app.route('/fl', methods=['POST'])
def run_fl():
    try:
        X = data1.iloc[20000:35000, :-1]
        y = data1.iloc[20000:35000, -1] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)

        client_data_sizes = [15000, 15000] 
        models_to_use = models_SVC[:2] 
        fl_results = fl.federated_learning_SVC(models_to_use, client_data_sizes, X_test, y_test, np.unique(y))
        models_SVC.clear()
        return jsonify(fl_results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
