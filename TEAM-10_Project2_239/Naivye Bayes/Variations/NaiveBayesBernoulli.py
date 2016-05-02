import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# reading csv file using pandas
df = pd.read_csv("/Users/balwindersingh/Desktop/crime_locationdescription.csv")

df_mod = df.copy()
targets = df_mod["crime_type"].unique()
# Mapping crime type labels to integer values
map_to_int = {name: n for n, name in enumerate(targets)}
df_mod["Target"] = df_mod["crime_type"].replace(map_to_int)


# Mapping 
location_type = df_mod["locationDesc"].unique()
map_location_to_int = {location: l for l, location in enumerate(location_type)}
df_mod["locationDesc"] = df_mod["locationDesc"].replace(map_location_to_int)


columns = ['hour', 'locationDesc']
df_filter = pd.DataFrame(columns=columns)
df_filter["hour"] = df_mod["hour"]
df_filter["district"] = df_mod["district"]
df_filter["locationDesc"] = df_mod["locationDesc"]
features = list(df_filter.columns[0:])


X = df_filter[features]
y=df_mod["Target"]


# splitting into test and training data


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state=0)
print "Creating classifier"
# using GaussianNB Theorem
clf =  BernoulliNB()
clf.fit(X_train, y_train)
print "predicting"
# predicting target class
y_pred = clf.predict(X_test)
#nb = clf.predict([[1294691400,41.7281217200,-87.6579890300]])


# calculating accuracy
print "Score is: ", metrics.accuracy_score(y_test,y_pred)

"""
scores = cross_val_score(clf,X,y,cv=10, scoring='accuracy')
print "Score: ",scores
"""