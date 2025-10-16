import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Iris.csv', index_col='Id')
df.dropna(inplace=True)

#Split data into Target and variables
X = df.drop(columns='Species')
Y = df['Species']

#Encode the target
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

#split data and train models
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

models = [KNeighborsClassifier(), LogisticRegression()]

for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    score = model.score(x_test,y_test)
    accuracy = accuracy_score(y_test,y_pred)
    model_name = str(model)[:-2]
    
    print(f"{model_name}")
    print(f"Model Accuracy: {score}")
    print(f"Perforance in percentage: {accuracy*100}\n")