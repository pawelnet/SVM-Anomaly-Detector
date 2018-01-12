import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing

def convertTextValues (train, test):
    test_services = test.service.unique()
    train_servics = train.service.unique()
    services = list(dict.fromkeys([*test_services, *train_servics]))
    le_service = preprocessing.LabelEncoder()
    le_service.fit(services)
    train.service = le_service.transform(train.service)
    test.service = le_service.transform(test.service)
    test_flag = test.flag.unique()
    train_flag = train.flag.unique()
    flags = list(dict.fromkeys([*test_flag, *train_flag]))
    le_flag = preprocessing.LabelEncoder()
    le_flag.fit(flags)
    train.flag = le_flag.transform(train.flag)
    test.flag = le_flag.transform(test.flag)

    test_protocol = test.protocol_type.unique()
    train_protocol = train.protocol_type.unique()
    protocols = list(dict.fromkeys([*test_protocol, *train_protocol]))
    le_protocol = preprocessing.LabelEncoder()
    le_protocol.fit(protocols)
    train.protocol_type = le_protocol.transform(train.protocol_type)
    test.protocol_type = le_protocol.transform(test.protocol_type)
    train.ix[train.label != "normal.", 'label'] = -1
    train.ix[train.label == "normal.", 'label'] = 1
    test.ix[test.label != "normal.", 'label'] = -1
    test.ix[test.label == "normal.", 'label'] = 1

    return train, test

def normalizeFeatures (dataset, features):
    for feature in features:
        dataset[feature] = np.log((dataset[feature] + 0.1).astype(float))

    return dataset

def getNuAndStripLabelFromTrainDataset(trainDataset):
    labels = train_dataset.label
    attacks = labels[labels == -1]
    print("attacks", attacks.shape[0])
    print("all", labels.shape[0])

    nu = attacks.shape[0] / labels.shape[0]
    if nu == 0.0:
        nu = 0.0001
    print("nu", nu)
    trainDataset.drop(["label"], axis=1, inplace=True)
    return nu, trainDataset

def prepareTestDataset(testDataset):
    test_label = test_dataset.label
    testDataset.drop(["label"], axis=1, inplace=True)
    return test_label, testDataset

print("--- Loading Datasets ---")
column_names = ["duration","protocol_type","service","flag","zsrc_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
train_dataset = pd.read_csv('kddcup.data_10_percent_corrected', names = column_names, index_col = False)
test_dataset = pd.read_csv('corrected', names = column_names, index_col = False)

# Wybranie tylko ruchu UDP
train_dataset = train_dataset.loc[train_dataset["protocol_type"] == "udp"]
test_dataset = test_dataset.loc[test_dataset["protocol_type"] == "udp"]

# Wybranie zbioru ruchu oznaczonego jako normal
train_dataset = train_dataset.loc[train_dataset["label"] == "normal."]

# Okreslenie istotnych atrybutow
features = ["protocol_type","service","flag","zsrc_bytes","wrong_fragment","srv_count","same_srv_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","label"]
features_no_label = ["service","flag","zsrc_bytes","wrong_fragment","srv_count","same_srv_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate"]
train_dataset = train_dataset[features]
test_dataset = test_dataset[features]

print("--- Preparing data for usage ---")

# Zakodowanie etykiet tekstowych
train_dataset, test_dataset = convertTextValues(train_dataset, test_dataset)

# Generacja nu oraz usuniecie nadzoru
nu, train_dataset = getNuAndStripLabelFromTrainDataset(train_dataset)

# Przygotowanie danych testowych
test_preds, test_dataset = prepareTestDataset(test_dataset)

# Normalizacja danych
train_dataset = normalizeFeatures(train_dataset, features_no_label)
test_dataset = normalizeFeatures(test_dataset, features_no_label)

# Nauka modelu oraz weryfikacja danych
model = svm.OneClassSVM(nu=nu, kernel='rbf', gamma=0.005, cache_size=1024)
print("--- Training Model ---")
model.fit(train_dataset)
print("--- Predicting Model ---")
preds = model.predict(test_dataset)
targs = test_preds
print("--- Results: ---")
print("accuracy: ", metrics.accuracy_score(targs, preds))
print("precision: ", metrics.precision_score(targs, preds))
print("recall: ", metrics.recall_score(targs, preds))
print("f1: ", metrics.f1_score(targs, preds))
print("area under curve (auc): ", metrics.roc_auc_score(targs, preds))