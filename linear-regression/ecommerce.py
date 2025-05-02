from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

# loading data
df = pd.read_csv("ecommerce.csv")

# features and target
X = df[['Time on App', 'Time on Website', 'Avg. Session Length', 'Length of Membership']]
y = df['Yearly Amount Spent']

# scaling the features (feature scaling needed for gradient descent)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sgd_model = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='invscaling', penalty=None, random_state=42)

#  cross-val
cv_scores = cross_val_score(sgd_model, X_scaled, y, cv=5, scoring='r2')

print("Cross-Validation R^2 scores:", cv_scores)
print("Average R^2 score:", np.mean(cv_scores))

# save the model and scaler
joblib.dump(sgd_model, 'sgd_regressor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

