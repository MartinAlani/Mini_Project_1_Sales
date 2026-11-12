import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("Loading....")
df = pd.read_csv('/home/malani/Desktop/Mini Project/sales - sales.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("Processing data")
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

df['state_holiday'] = df['state_holiday'].astype(str)
df = pd.get_dummies(df,columns=['state_holiday'],drop_first=True)

features = [
    'store_ID','day_of_week','nb_customers_on_day',
    'open', 'promotion','school_holiday','year','month','day'
]

holiday_cols =[col for col in df.columns if 'state_holiday_' in col]
features.extend(holiday_cols)

X = df[features]
y = df['sales']

print("Spliting data set into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

print("Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=1
)
rf_model.fit(X_train, y_train)

print("Evaluating model performance")
y_pred = rf_model.predict(X_test)

rms = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--Model Results--")
print(f"Root Mean Squared Error(RMSE): {rms:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print(importances.head(10))