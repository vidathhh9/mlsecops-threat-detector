import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'
]

def load_and_preprocess(train_path, test_path):
    print("Loading datasets...")
    train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=COLUMNS)

    train_df = train_df.drop('level', axis=1)
    test_df = test_df.drop('level', axis=1)

    train_df['is_attack'] = train_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)
    test_df['is_attack'] = test_df['attack'].apply(lambda x: 0 if x == 'normal' else 1)

    train_df = train_df.drop('attack', axis=1)
    test_df = test_df.drop('attack', axis=1)

    categorical_cols = ['protocol_type', 'service', 'flag']
    combined = pd.concat([train_df, test_df], axis=0)
    combined = pd.get_dummies(combined, columns=categorical_cols)

    train_encoded = combined.iloc[:len(train_df)].copy()
    test_encoded = combined.iloc[len(train_df):].copy()

    X_train = train_encoded.drop('is_attack', axis=1)
    y_train = train_encoded['is_attack'].astype(int)

    X_test = test_encoded.drop('is_attack', axis=1)
    y_test = test_encoded['is_attack'].astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler

def train_model():
    train_path = os.path.join('data', 'KDDTrain+.txt')
    test_path = os.path.join('data', 'KDDTest+.txt')

    X_train, y_train, X_test, y_test, scaler = load_and_preprocess(train_path, test_path)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/threat_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Saved model and scaler to models/")

if __name__ == '__main__':
    train_model()