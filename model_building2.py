import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
import pickle

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
df = pd.read_csv('eda_data.csv')

# Select relevant columns
df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry',
               'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided',
               'job_state', 'same_state', 'age', 'python_yn', 'spark', 'aws',
               'excel', 'job_simp', 'seniority', 'desc_len']]

# -------------------------------------------------------------
# 2. CREATE DUMMIES AND CLEAN DATA
# -------------------------------------------------------------
df_dum = pd.get_dummies(df_model, drop_first=True)

# Remove any remaining non-numeric or missing values
df_dum = df_dum.apply(pd.to_numeric, errors='coerce')
df_dum = df_dum.fillna(0)

# -------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# -------------------------------------------------------------
X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# 4. MULTIPLE LINEAR REGRESSION (statsmodels)
# -------------------------------------------------------------
X_sm = sm.add_constant(X)  # ✅ FIXED: don't overwrite X with add_constant
model = sm.OLS(y, X_sm.astype(float))  # ✅ ensure numeric dtype
results = model.fit()
print(results.summary())

# -------------------------------------------------------------
# 5. LINEAR REGRESSION (SKLEARN)
# -------------------------------------------------------------
lm = LinearRegression()
lm.fit(X_train, y_train)
print("Linear Regression CV MAE:",
      np.mean(cross_val_score(lm, X_train, y_train, scoring='mean_absolute_error', cv=3)))

# -------------------------------------------------------------
# 6. LASSO REGRESSION
# -------------------------------------------------------------
lm_l = Lasso(alpha=0.13)
lm_l.fit(X_train, y_train)
print("Lasso CV MAE:",
      np.mean(cross_val_score(lm_l, X_train, y_train, scoring='mean_absolute_error', cv=3)))

# -------------------------------------------------------------
# 7. FIND BEST LASSO ALPHA
# -------------------------------------------------------------
alpha = []
error = []

for i in range(1, 100):
    a = i / 100
    lml = Lasso(alpha=a)
    err = np.mean(cross_val_score(lml, X_train, y_train, scoring='mean_absolute_error', cv=3))
    alpha.append(a)
    error.append(err)

plt.plot(alpha, error)
plt.title("Lasso Alpha vs Error")
plt.xlabel("Alpha")
plt.ylabel("Negative MAE")
plt.show()

df_err = pd.DataFrame({'alpha': alpha, 'error': error})
print("Best Alpha:", df_err.loc[df_err['error'].idxmax()])

# -------------------------------------------------------------
# 8. RANDOM FOREST
# -------------------------------------------------------------
rf = RandomForestRegressor(random_state=42)
print("Random Forest CV MAE:",
      np.mean(cross_val_score(rf, X_train, y_train, scoring='mean_absolute_error', cv=3)))

# -------------------------------------------------------------
# 9. GRID SEARCH CV FOR RANDOM FOREST
# -------------------------------------------------------------
parameters = {
    'n_estimators': range(10, 300, 10),
    'criterion': ('squared_error', 'absolute_error'),
    'max_features': ('auto', 'sqrt', 'log2')
}

gs = GridSearchCV(rf, parameters, scoring='mean_absolute_error', cv=3, n_jobs=-1)
gs.fit(X_train, y_train)

print("Best GridSearchCV Score:", gs.best_score_)
print("Best Estimator:", gs.best_estimator_)

# -------------------------------------------------------------
# 10. TEST ENSEMBLES
# -------------------------------------------------------------
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

print("MAE Linear:", mean_absolute_error(y_test, tpred_lm))
print("MAE Lasso:", mean_absolute_error(y_test, tpred_lml))
print("MAE RF:", mean_absolute_error(y_test, tpred_rf))
print("MAE Ensemble:", mean_absolute_error(y_test, (tpred_lm + tpred_rf) / 2))

# -------------------------------------------------------------
# 11. SAVE MODEL
# -------------------------------------------------------------
pickl = {'model': gs.best_estimator_}
pickle.dump(pickl, open('model_file.p', "wb"))

# -------------------------------------------------------------
# 12. LOAD MODEL AND TEST PREDICTION
# -------------------------------------------------------------
with open('model_file.p', 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

sample_pred = model.predict(np.array(list(X_test.iloc[1, :])).reshape(1, -1))
print("Sample Prediction:", sample_pred)

