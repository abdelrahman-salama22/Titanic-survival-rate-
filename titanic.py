import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')


df = pd.read_csv('D:/ai course/Session 16_ Logistic Regression/titanic.csv')
#print(df.info())
#print (df.head())
#-------------Data preprocessing----------------
# Drop unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin','Fare'], axis=1, inplace=True)
# Handle missing values
#print( df.isnull().sum())
def handling_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return 37
        elif row['Pclass'] == 2:
            return 29
        else:
            return 24
    else:
        return row['Age'] 
    
df['Age'] = df.apply(handling_age, axis=1)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print (df.head())
# encoding the categorical variables
encoder = LabelEncoder()
df ['Sex']=encoder.fit_transform(df['Sex'])
df ['Embarked']=encoder.fit_transform(df['Embarked'])
#print (df.head())
# scaling the featuers 
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
#print (df.head())
#spliting the data into features and target 
X = df.drop('Survived', axis=1)
y = df['Survived']
#spli the data into taring and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#-------------Model training----------------
model = LogisticRegression()
model.fit(X_train, y_train)
#-------------Model evaluation----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
# ---------------- Visualization ----------------

# choose only 2 features for visualization
X_vis = df[['Age', 'Pclass']]
y_vis = df['Survived']

# split again for visualization
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis, y_vis, test_size=0.2, random_state=42
)

# train new model on 2 features only
model_vis = LogisticRegression()
model_vis.fit(X_train_vis, y_train_vis)

# create meshgrid
x_min, x_max = X_vis.iloc[:, 0].min() - 1, X_vis.iloc[:, 0].max() + 1
y_min, y_max = X_vis.iloc[:, 1].min() - 1, X_vis.iloc[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# plot decision boundary
plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=y_vis, edgecolors='k')

plt.xlabel("Age (Scaled)")
plt.ylabel("Pclass")
plt.title("Decision Boundary - Logistic Regression (Titanic)")
plt.show()
