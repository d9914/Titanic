import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
train=pd.read_csv("train.csv")

# Remove unnecessary columns
train = train.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing values
median_age = train['Age'].median()
train['Age'].fillna(median_age, inplace=True)

most_common_embarked = train['Embarked'].value_counts().index[0]
train['Embarked'].fillna(most_common_embarked, inplace=True)

# Perform feature engineering
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train['FarePerPerson'] = train['Fare'] / train['FamilySize']

bins = [0, 18, 65, 120]
labels = ['0', '1', '2']
train['Age_Category'] = pd.cut(train['Age'], bins=bins, labels=labels, include_lowest=True)

# Encode categorical features
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])

embarked_one_hot = pd.get_dummies(train['Embarked'], prefix='Embarked')
train = pd.concat([train, embarked_one_hot], axis=1)
train = train.drop('Embarked', axis=1)

# Split the data into training and testing sets
X = train.drop(['Survived'], axis=1)
y = train['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

# Evaluate model accuracy
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)


# Convert predictions to a dataframe
predictions_df = pd.DataFrame({'PassengerId': X_test['PassengerId'], 'Survived': y_pred})

# Save dataframe to a CSV file
predictions_df.to_csv('predictions.csv', index=False)