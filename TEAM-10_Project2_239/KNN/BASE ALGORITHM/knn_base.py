import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

## Variations------
# no of record: 15 lakhs records (full dataset)
## Value of K = 5 (default)
## distance metric: 'Minkowski' (default)
## train test split : 0.75/0.25 (default)

df = pd.read_csv("/Users/rainashastri/Desktop/TEAM-10_Project2_239/KNN/BASE ALGORITHM/district_with_loc_id-full.csv")

df_mod = df.copy()
targets = df_mod["crime_type"].unique()
# Mapping crime type labels to integer values
map_to_int = {name: n for n, name in enumerate(targets)}
df_mod["Target"] = df_mod["crime_type"].replace(map_to_int)

features = list(df_mod.columns[1:5])
X = df_mod[features]
y=df_mod["Target"]

#Spliting into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf =  KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test,y_pred)
print "Score is: ", score
