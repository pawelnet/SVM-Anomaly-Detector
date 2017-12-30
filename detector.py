import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Ladowanie danych
column_names = ["duration","protocol_type","service","flag","zsrc_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]
dataset = pd.read_csv('kddcup.data_10_percent_corrected', names = column_names, index_col = False)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#zakodowanie stalych
for i in range(1, 4):
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    
onehotencoder = OneHotEncoder(categorical_features = [1,2,3])
X = onehotencoder.fit_transform(X).toarray()

#Podzial zbioru
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size = 0.01, random_state = 0)


# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_dev)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_dev)

n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size


