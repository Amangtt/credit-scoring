import pandas as pd
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import json
import argparse

def evaluate(input_path,model_path,metrics_path):
    logging.info('model evaluation calculating selected metrics....')
    df=pd.read_csv(input_path)
    model= joblib.load(model_path)
    col = ['Risk_Label', 'CustomerId','WoE']
    x = df.drop(col, axis=1)
    y = df['Risk_Label']
   
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Evaluate the model

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2=r2_score(y_test, y_pred)
    metrics = {
        'mean_squared_error': mse,
        'r2_score': r2
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help='Path to read csv')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--output', required=True, help='Path to output metrics file')
    args=parser.parse_args()
    evaluate(args.input,args.model,args.output)