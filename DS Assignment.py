df = pd.read_excel(r'DS - Assignment Part 1 data set.xlsx')
df.head()

df.isnull().sum()

df['Transaction date'].unique()

df.corr()

df.shape

df.describe()

X = df.drop(['TARGET(House price of unit area)'], axis = 1)
X

y = df['TARGET(PRICE_IN_LACS)']
y

X = pd.get_dummies(X, columns = ['POSTED_BY', 'BHK_OR_RK'], drop_first = True)
X

scaler = MinMaxScaler()
X_scaler = scaler.fit_transform(X)
X_scaler


df.shape




X_train, X_test, y_train, y_test=train_test_split(X_scaler, y,  test_size = 0.3, random_state = 1)



lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


lr_model.score(X_train, y_train)

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)


knn_model.score(X_test, y_test)

knn_model.score(X_train, y_train)


dt_model = DecisionTreeRegressor(max_depth = 5)
dt_model.fit(X_train, y_train)


dt_model.score(X_test, y_test)



dt_model.score(X_train, y_train)




testdf = pd.read_csv(r'test.csv')
testdf.head()



testdf.isnull().sum()


Xtest = testdf.drop(['ADDRESS'], axis = 1)
Xtest


Xtest_encoded = pd.get_dummies(Xtest, columns = ['POSTED_BY', 'BHK_OR_RK'], drop_first = True)
Xtest_encoded



Xtest_encoded_scaled = scaler.transform(Xtest_encoded)
Xtest_encoded_scaled



test_y_pred = dt_model.predict(Xtest_encoded_scaled)


test_y_result_df = pd.DataFrame(test_y_pred, columns = ['TARGET(PRICE_IN_LACS)'])
test_y_result_df



test_y_result_df.to_csv(r'result.csv', index = None)


