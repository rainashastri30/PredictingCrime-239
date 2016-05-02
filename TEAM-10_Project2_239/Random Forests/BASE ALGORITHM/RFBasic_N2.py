import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

df = pd.read_csv("F:/dataset239/final/2011_16_district.csv")

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
features = list(df_filter.columns[1:4])

X = df_filter[features]
y=df_mod["Target"]

print("Random Forest begins")


rf=RandomForestClassifier(min_samples_leaf=2)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)

# X_train, X_test, y_train, y_test = train_test_split(X, y)
print (X_train.shape)
print (X_test.shape)
print (y_train.shape)
print (y_test.shape)
rf=rf.fit(X_train, y_train)

# STEP 3: make predictions on the testing set
y_pred = rf.predict(X_test)

# compare actual response values (y_test) with predicted response values (y_pred)
print (metrics.accuracy_score(y_test, y_pred))

print (mean_squared_error(y_test, y_pred))
#rfans = rf.predict([[1294691400,41.7281217200,-87.6579890300]])


#scores = cross_val_score(rf,X,y,cv=10, scoring='accuracy')
#print ("Score: ",scores)

print ("Random forest ends")