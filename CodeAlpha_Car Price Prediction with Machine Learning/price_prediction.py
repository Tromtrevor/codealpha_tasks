import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as date
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('car data.csv')
df= df.drop_duplicates()

year_now = date.now().year
df['Age'] = year_now-df['Year']

#split data to target and variables
X = df.drop(columns=['Year', 'Selling_Price'])
Y = df['Selling_Price']

#encoding
cols_to_encode = X.select_dtypes(include='object').columns
encoders = {}
for col in cols_to_encode:
    encoder = LabelEncoder()
    X[col] = encoder.fit_transform(X[col])
    encoders[col] = encoder


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3,random_state=42)
models = {'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42, n_estimators=500, max_depth=2, min_samples_split=2), 
          'XGBoost Regressor': XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.1, max_depth=2, colsample_bytree=0.8, subsample=0.8, n_jobs=-1),
          'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=500, max_depth=6, min_samples_split=2, min_samples_leaf=1)
         }
#fig,ax  = plt.subplots(1,3,figsize=(10,4))
count=0
scores ={}
for name,model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)

    # sns.scatterplot(x=y_test,y=y_pred, ax=ax[count])
    # ax[count].set_xlabel('Actual')
    # ax[count].set_ylabel('Predictions')
    # ax[count].set_title(name)
    # count+=1
    scores[name]=score

highest_score = max(scores, key=scores.get)
final_model = models[highest_score]

print(f"Best model\n{highest_score}: {scores[highest_score]}")
print("\nOther models")
for item in scores:
    if item != highest_score:
        print(f"{item}: {scores[item]}")

plt.show()

joblib.dump(encoders, 'model/encoders.pkl')
joblib.dump(final_model, 'model/price_prediction_model.pkl')

print(f'\n✔ Model and encoders saved successfully.')