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

# ---------- Configuration ----------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = None
scaler_X = None
scaler_y = None
feature_names = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_plots(y_true, y_pred, loss_curve, stopped_epoch):
    # Loss curve
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

    # Scatter plot
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_true, y_pred, alpha=0.6)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
    ax2.set_xlabel('Observed (mm)')
    ax2.set_ylabel('Predicted (mm)')
    ax2.set_title('Observed vs Predicted')
    ax2.legend()
    ax2.grid(True)
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    scatter_b64 = base64.b64encode(buf2.read()).decode()
    plt.close(fig2)

    # Time series
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    indices = range(len(y_true))
    ax3.plot(indices, y_true, 'b-', label='Observed')
    ax3.plot(indices, y_pred, 'r-', label='Predicted')
    ax3.set_xlabel('Test sample index')
    ax3.set_ylabel('Rainfall (mm)')
    ax3.set_title('Time Series on Test Set')
    ax3.legend()
    ax3.grid(True)
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png')
    buf3.seek(0)
    ts_b64 = base64.b64encode(buf3.read()).decode()
    plt.close(fig3)

    return loss_b64, scatter_b64, ts_b64

# ---------- HTML Templates ----------
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Rainfall Forecasting - ANFIS</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        h1 { color: #2c3e50; }
        .section {
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .feature-group {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        input, button {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        button {
            background: #28a745;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #218838; }
        .info { background: #e7f3ff; border-left: 4px solid #2196F3; padding: 10px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌧️ Rainfall Forecasting using ANFIS</h1>
        <p>Upload Excel, choose features, set max epochs. Early stopping will stop when validation loss plateaus.</p>
        <form action="/train" method="post" enctype="multipart/form-data">
            <label>📁 Excel File (.xlsx):</label>
            <input type="file" name="file" accept=".xlsx" required><br>

            <div class="section">
                <strong>🎛️ Rainfall lags (1-12 months):</strong>
                <div class="feature-group">
                    {% for i in range(1,13) %}
                    <div><input type="checkbox" name="features" value="Rain_t{{i}}" id="lag{{i}}">
                    <label for="lag{{i}}">Rain_t{{i}}</label></div>
                    {% endfor %}
                </div>
            </div>

            <div class="section">
                <strong>🌡️ Meteorological variables (same month):</strong>
                <div class="feature-group">
                    <div><input type="checkbox" name="features" value="RH2"> <label>RH2</label></div>
                    <div><input type="checkbox" name="features" value="SR"> <label>SR</label></div>
                    <div><input type="checkbox" name="features" value="WS"> <label>WS</label></div>
                    <div><input type="checkbox" name="features" value="Tmax"> <label>Tmax</label></div>
                </div>
            </div>

            <label>⚙️ Max Epochs:</label>
            <input type="number" name="max_epochs" value="300" min="10" max="1000" step="10"><br>
            <label>⏸️ Early stopping patience:</label>
            <input type="number" name="patience" value="20" min="5" max="100" step="5"><br>
            <button type="submit">🚀 Train Model</button>
        </form>
        <div class="info">BRIJITH ANITHA T M.Sc Agricultural Statistics.,@TNAU.</div>
    </div>
</body>
</html>
"""

train_result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Complete</title>
    <style>
        body { font-family: Arial; background: #f4f4f4; text-align: center; padding: 50px; }
        .card { background: white; border-radius: 20px; padding: 30px; max-width: 900px; margin: auto; }
        .metrics { background: #e9ecef; border-radius: 10px; padding: 15px; margin: 20px 0; }
        img { max-width: 100%; border-radius: 10px; margin: 10px 0; }
        a { background: #007bff; color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; display: inline-block; margin: 10px; }
        .early-stop { background: #d4edda; padding: 10px; border-radius: 8px; }
    </style>
</head>
<body>
    <div class="card">
        <h2>✅ Training Completed</h2>
        <div class="early-stop">🛑 Early stopping at epoch {{ stopped_epoch }} (patience={{ patience }})</div>
        <div class="metrics">
            <p><strong>Features:</strong> {{ features }}</p>
            <p><strong>Actual epochs used:</strong> {{ actual_epochs }} / {{ max_epochs }}</p>
            <p><strong>Test RMSE:</strong> {{ rmse }} mm &nbsp;|&nbsp; <strong>R²:</strong> {{ r2 }}</p>
        </div>
        <img src="data:image/png;base64,{{ loss_plot }}" alt="Loss Curve">
        <img src="data:image/png;base64,{{ scatter_plot }}" alt="Scatter">
        <img src="data:image/png;base64,{{ timeseries_plot }}" alt="Time Series">
        <br>
        <a href="/predict_page">🔮 Go to Prediction</a>
        <a href="/">🏠 Retrain</a>
    </div>
</body>
</html>
"""

predict_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Predict</title>
    <style>
        body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }
        .container { max-width: 600px; margin: auto; background: white; border-radius: 20px; padding: 30px; }
        input { width: 90%; padding: 10px; margin: 8px 0; border-radius: 8px; }
        button { background: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 25px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Predict Next Month's Rainfall</h1>
        <form action="/predict" method="post">
            {% for f in features %}
            <label>{{ f }} (mm or original unit):</label>
            <input type="number" step="any" name="{{ f }}" required><br>
            {% endfor %}
            <button type="submit">Predict</button>
        </form>
    </div>
</body>
</html>
"""

predict_result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Result</title>
    <style>
        body { font-family: Arial; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; }
        .container { max-width: 500px; margin: auto; background: white; border-radius: 20px; padding: 30px; text-align: center; }
        .result { font-size: 3em; font-weight: bold; color: #28a745; }
        a { display: inline-block; background: #007bff; color: white; text-decoration: none; padding: 10px 20px; border-radius: 25px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 Forecast</h1>
        <div class="result">{{ prediction }} mm</div>
        <p>predicted rainfall for next month</p>
        <a href="/predict_page">← New</a> <a href="/">🏠 Home</a>
    </div>
</body>
</html>
"""

# ---------- Flask Routes ----------
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

        mlp_model = MLPRegressor(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            max_iter=max_epochs,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=patience,
            random_state=42
        )
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

        return render_template_string(
            train_result_html,
            features=', '.join(features),
            max_epochs=max_epochs,
            actual_epochs=len(loss_curve),
            stopped_epoch=stopped_epoch,
            patience=patience,
            rmse=round(rmse, 2),
            r2=round(r2, 4),
            loss_plot=loss_b64,
            scatter_plot=scatter_b64,
            timeseries_plot=ts_b64
        )
    finally:
        os.remove(filepath)

@app.route('/predict_page')
def predict_page():
    if model is None:
        return "Model not trained yet. Please train a model first."
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
    app.run(host="0.0.0.0", port=10000)
