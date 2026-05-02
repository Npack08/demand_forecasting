from flask import Flask, render_template, request, redirect, url_for, flash
import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

DATABASE = 'database.db'

def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY,
            name TEXT,
            model_path TEXT,
            metrics TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            mae REAL,
            rmse REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

init_db()

@app.route('/')
def dataset_view():
    try:
        df = pd.read_csv('demand_forecasting.csv')
        num_rows = len(df)
        num_cols = len(df.columns)
        column_names = df.columns.tolist()
        missing_values = df.isnull().sum().to_dict()
        data_types = df.dtypes.astype(str).to_dict()
        preview = df.head(50).to_html()
        return render_template('dataset.html', preview=preview, num_rows=num_rows, num_cols=num_cols, column_names=column_names, missing_values=missing_values, data_types=data_types)
    except Exception as e:
        flash(f'Error loading dataset: {str(e)}')
        return render_template('dataset.html', preview=None)

def preprocess_data(df, encoders=None):
    # Select features and target
    features = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Promotion', 'Competitor Pricing', 'Epidemic']
    categorical = ['Category', 'Region', 'Weather Condition', 'Seasonality']
    target = 'Demand'
    
    # Encode categoricals
    le_dict = {}
    for col in categorical:
        if encoders and col in encoders:
            df[col] = encoders[col].transform(df[col])
            le_dict[col] = encoders[col]
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        features.append(col)
    
    X = df[features]
    y = df[target] if target in df.columns else None
    return X, y, features, le_dict

@app.route('/train', methods=['GET', 'POST'])
def train():
    training_result = None
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        if not model_type:
            flash('Please select a model type.')
            return render_template('train.html', training_result=training_result)
        
        try:
            df = pd.read_csv('demand_forecasting.csv')
            X, y, features, _ = preprocess_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'rf':
                model = RandomForestRegressor(random_state=42)
            else:
                flash('Invalid model type.')
                return render_template('train.html', training_result=training_result)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            os.makedirs('models', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f'{model_type}_{timestamp}'
            model_path = os.path.join('models', f'{model_name}.joblib')
            joblib.dump(model, model_path)
            
            metrics = json.dumps({'mae': mae, 'rmse': rmse, 'features': features})
            with get_db() as conn:
                conn.execute('INSERT INTO models (name, model_path, metrics) VALUES (?, ?, ?)', (model_name, model_path, metrics))
                conn.execute('INSERT INTO experiments (model_name, mae, rmse) VALUES (?, ?, ?)', (model_name, mae, rmse))
            flash(f'Training complete! Model saved as {model_name}.', 'success')
            return redirect(url_for('results'))
        except Exception as e:
            flash(f'Error during training: {str(e)}', 'danger')
    return render_template('train.html', training_result=training_result)

@app.route('/results')
def results():
    try:
        with get_db() as conn:
            cursor = conn.execute(
                'SELECT e.model_name, e.mae, e.rmse, e.created_at, m.model_path '
                'FROM experiments e '
                'LEFT JOIN models m ON e.model_name = m.name '
                'ORDER BY e.created_at DESC'
            )
            rows = cursor.fetchall()
        experiments = [
            {
                'model_name': row[0],
                'mae': row[1],
                'rmse': row[2],
                'created_at': row[3],
                'model_path': row[4]
            }
            for row in rows
        ]
        best_rmse = min((exp['rmse'] for exp in experiments), default=None)
        return render_template('results.html', experiments=experiments, best_rmse=best_rmse)
    except Exception as e:
        flash(f'Error loading results: {str(e)}')
        return render_template('results.html', experiments=None, best_rmse=None)

@app.route('/registry')
def registry():
    try:
        with get_db() as conn:
            rows = conn.execute('SELECT id, name, model_path, metrics, created_at FROM models ORDER BY created_at DESC').fetchall()
        models = []
        for row in rows:
            metrics = json.loads(row[3]) if row[3] else {}
            models.append({
                'id': row[0],
                'name': row[1],
                'model_path': row[2],
                'metrics': {
                    'mae': metrics.get('mae'),
                    'rmse': metrics.get('rmse'),
                    'features': metrics.get('features', [])
                },
                'created_at': row[4]
            })
        return render_template('registry.html', models=models)
    except Exception as e:
        flash(f'Error loading registry: {str(e)}', 'danger')
        return render_template('registry.html', models=None)

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    try:
        with get_db() as conn:
            models = conn.execute('SELECT name FROM models ORDER BY created_at DESC').fetchall()
    except:
        models = []

    try:
        df = pd.read_csv('demand_forecasting.csv')
        product_ids = sorted(df['Product ID'].dropna().unique().tolist())
    except Exception:
        product_ids = []

    selected_model = models[0][0] if models else None
    selected_product = product_ids[0] if product_ids else None

    if request.method == 'POST':
        selected_model = request.form.get('model_name') or selected_model
        selected_product = request.form.get('product_id') or selected_product
        sample_input = {
            'Inventory Level': 150,
            'Units Sold': 100,
            'Units Ordered': 200,
            'Price': 50.0,
            'Discount': 5,
            'Promotion': 1,
            'Competitor Pricing': 55.0,
            'Epidemic': 0,
            'Category': 'Electronics',
            'Region': 'North',
            'Weather Condition': 'Sunny',
            'Seasonality': 'Summer',
            'Product ID': selected_product,
            'Store ID': 'S001'
        }

        try:
            with get_db() as conn:
                model_row = conn.execute('SELECT model_path, metrics FROM models WHERE name = ?', (selected_model,)).fetchone()
            if not model_row:
                flash('Model not found.')
                return redirect(url_for('inference'))

            model_path = model_row[0]
            metrics = json.loads(model_row[1])
            features = metrics['features']

            model = joblib.load(model_path)

            train_df = pd.read_csv('demand_forecasting.csv')
            _, _, _, encoders = preprocess_data(train_df)
            df_sample = pd.DataFrame([sample_input])
            _, _, _, _ = preprocess_data(df_sample, encoders=encoders)
            X_sample = df_sample[features]

            prediction = model.predict(X_sample)[0]
            flash(f'Predicted Demand for {selected_product}: {prediction:.2f}')
            return redirect(url_for('inference'))
        except Exception as e:
            flash(f'Error during inference: {str(e)}')
            return redirect(url_for('inference'))

    return render_template('inference.html', models=models, product_ids=product_ids, selected_model=selected_model, selected_product=selected_product)

@app.route('/model_predictions/<model_name>')
def model_predictions(model_name):
    product_id = request.args.get('product_id')
    if not product_id:
        return {'error': 'product_id query parameter is required'}, 400

    try:
        with get_db() as conn:
            model_row = conn.execute('SELECT model_path, metrics FROM models WHERE name = ?', (model_name,)).fetchone()
        if not model_row:
            return {'error': 'Model not found'}, 404

        model_path = model_row[0]
        metrics = json.loads(model_row[1])
        features = metrics['features']
        model = joblib.load(model_path)

        df = pd.read_csv('demand_forecasting.csv')
        if 'Product ID' not in df.columns:
            return {'error': 'Dataset does not contain Product ID'}, 500

        product_df = df[df['Product ID'] == product_id].copy()
        if product_df.empty:
            return {'error': f'No data for product_id {product_id}'}, 404

        product_df['Date'] = pd.to_datetime(product_df['Date'])
        product_df = product_df.sort_values('Date').tail(30).reset_index(drop=True)

        sample_input = {
            'Product ID': product_id,
            'Category': product_df.iloc[-1]['Category'],
            'Store ID': product_df.iloc[-1]['Store ID'],
            'Region': product_df.iloc[-1]['Region']
        }

        df_predict = product_df.copy()
        df_predict = df_predict.drop(columns=[col for col in ['Demand'] if col in df_predict.columns])
        _, _, _, encoders = preprocess_data(pd.read_csv('demand_forecasting.csv'))
        _, _, _, _ = preprocess_data(df_predict, encoders=encoders)
        X_predict = df_predict[features]

        predictions = model.predict(X_predict).tolist()
        dates = product_df['Date'].dt.strftime('%Y-%m-%d').tolist()

        return {
            'dates': dates,
            'predictions': predictions,
            'product_id': sample_input['Product ID'],
            'category': sample_input['Category'],
            'store_id': sample_input['Store ID'],
            'region': sample_input['Region']
        }
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)