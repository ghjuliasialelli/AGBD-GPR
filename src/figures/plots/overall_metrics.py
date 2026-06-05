from config import DATA_ROOT
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_metrics(y_true, y_pred) :

    me = np.mean(y_pred - y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"ME: {me:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")




# Our best
with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/nico_film/2020/17997535-1_17997535-2_17997535-3/47250199-density_test.pkl','rb') as f: ours = pickle.load(f)
y_true, y_pred = np.array(ours['ref']), np.array(ours['post'])
print("Our best model metrics:")
compute_metrics(y_true, y_pred)

# CCI
with open(f'{DATA_ROOT}/EcosystemAnalysis/Models/Biomes/kriging/predictions/CCI/CCI-density_test.pkl', 'rb') as f: cci = pickle.load(f)
y_true, y_pred = np.array(cci['ref']), np.array(cci['post'])
print("CCI model metrics:")
compute_metrics(y_true, y_pred)