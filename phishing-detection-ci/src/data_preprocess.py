'''import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler


df_train = pd.read_csv("phishing-detection-cicid/data/training.csv")
df_test = pd.read_csv('phishing-detection-cicid/data/testing.csv')

print("Training Data Shape:",df_train.shape)
print("Testing Data Shape:",df_test.shape)

print(df_train.head())

#drop url column
df_train.drop(columns=['url'],inplace =True)
df_test.drop(columns = ['url'],inplace = True) 

#status to numeric
le = LabelEncoder()
df_train['status'] = le.fit_transform(df_train['status'])

df_test['status'] = le.fit_transform(df_test['status'])

#separate labels

X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]

X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.dtypes)
print(y_train.dtypes)


__all__ = ['X_train', 'y_train', 'X_test', 'y_test']'''

def preprocess_data(file_path):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(file_path)
    df.drop(columns=['url'], inplace=True)

    le = LabelEncoder()
    df['status'] = le.fit_transform(df['status'])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y



