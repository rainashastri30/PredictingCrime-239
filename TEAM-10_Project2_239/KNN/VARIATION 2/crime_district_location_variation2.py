import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics



## Variations------
# no of record: 9 million records
## Value of K 60
## distance metric: 'manhattan'
## train test split : 60/40

df = pd.read_csv("/Users/rainashastri/Desktop/TEAM-10_Project2_239/KNN/VARIATION 2/district_with_loc_id-split-2.csv")

df_mod = df.copy()
targets = df_mod["crime_type"].unique()
# Mapping crime type labels to integer values
map_to_int = {name: n for n, name in enumerate(targets)}
df_mod["Target"] = df_mod["crime_type"].replace(map_to_int)

features = list(df_mod.columns[1:5])
X = df_mod[features]
y=df_mod["Target"]

# Spliting into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print "Creating classifier"
clf =  KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='manhattan',
          n_jobs=1, n_neighbors=60, p=1,
          weights='distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test,y_pred)
print "Score is: ", score
	

