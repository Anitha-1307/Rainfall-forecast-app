import pandas as pd
import numpy as np
import os
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

model = None
scaler_X = None
scaler_y = None
feature_names = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_plots(y_true, y_pred, loss_curve, stopped_epoch):
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(loss_curve, 'b-')
    ax1.axvline(x=stopped_epoch, color='r', linestyle='--', label=f'Early stop @ {stopped_epoch}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    loss_b64 = base64.b64encode(buf1.read()).decode()
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_true, y_pred, alpha=0.6)
    maxv, minv = max(y_true.max(), y_pred.max()), min(y_true.min(), y_pred.min())
    ax2.plot([minv, maxv], [minv, maxv], 'r--', label='Perfect')
    ax2.set_xlabel('Observed')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Observed vs Predicted')
    ax2.legend()
    ax2.grid(True)
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    scatter_b64 = base64.b64encode(buf2.read()).decode()
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(range(len(y_true)), y_true, 'b-', label='Observed')
    ax3.plot(range(len(y_pred)), y_pred, 'r-', label='Predicted')
    ax3.set_xlabel('Test sample')
    ax3.set_ylabel('Rainfall (mm)')
    ax3.set_title('Time Series')
    ax3.legend()
    ax3.grid(True)
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png')
    buf3.seek(0)
    ts_b64 = base64.b64encode(buf3.read()).decode()
    plt.close(fig3)
    return loss_b64, scatter_b64, ts_b64

# --- HTML Templates (You can keep the same as before) ---
# ... (place the index_html, train_result_html, predict_html, and predict_result_html strings here) ...

@app.route('/')
def home():
    return render_template_string(index_html)

@app.route('/train', methods=['POST'])
def train():
    global model, scaler_X, scaler_y, feature_names
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    if not allowed_file(file.filename):
        return "Invalid file type. Please upload an .xlsx file.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_excel(filepath)
        for lag in range(1, 13):
            df[f'Rain_t{lag}'] = df['Rainfall'].shift(lag)
        df_clean = df.dropna().reset_index(drop=True)

        features = request.form.getlist('features')
        if not features:
            return "No features selected", 400
        max_epochs = int(request.form.get('max_epochs', 300))
        patience = int(request.form.get('patience', 20))

        X = df_clean[features].values
        y = df_clean['Rainfall'].values
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_norm = scaler_X.fit_transform(X)
        y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_temp, X_test, y_temp, y_test = train_test_split(X_norm, y_norm, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)

        mlp_model = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', max_iter=max_epochs,
                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=patience, random_state=42)
        mlp_model.fit(X_train, y_train)

        loss_curve = mlp_model.loss_curve_
        stopped_epoch = mlp_model.n_iter_

        y_pred_norm = mlp_model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        loss_b64, scatter_b64, ts_b64 = generate_plots(y_true, y_pred, loss_curve, stopped_epoch)

        model = mlp_model
        scaler_X = scaler_X
        scaler_y = scaler_y
        feature_names = features

        return render_template_string(train_result_html, features=', '.join(features), max_epochs=max_epochs,
                                      actual_epochs=len(loss_curve), stopped_epoch=stopped_epoch, patience=patience,
                                      rmse=round(rmse, 2), r2=round(r2, 4), loss_plot=loss_b64,
                                      scatter_plot=scatter_b64, timeseries_plot=ts_b64)
    except Exception as e:
        return str(e), 500
    finally:
        os.remove(filepath)

@app.route('/predict_page')
def predict_page():
    if model is None:
        return "Train a model first"
    return render_template_string(predict_html, features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "No model trained", 400
    vals = [float(request.form[f]) for f in feature_names]
    X_input = np.array([vals])
    X_norm = scaler_X.transform(X_input)
    pred_norm = model.predict(X_norm)[0]
    pred = scaler_y.inverse_transform([[pred_norm]])[0][0]
    return render_template_string(predict_result_html, prediction=round(pred, 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)