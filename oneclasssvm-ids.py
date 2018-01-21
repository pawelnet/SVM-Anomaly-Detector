#author: gsiewruk, pkazmierczyk

import pandas as pd
import numpy as np
import sys
import math
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import time
start_time = time.time()

def convert_text_values (train, test):
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

def normalize_features (dataset, features):
    for feature in features:
        dataset[feature] = np.log((dataset[feature] + 0.1).astype(float))

    return dataset

def get_nu_and_strip_labels(trainDataset,nu):
    labels = train_dataset.label
    attacks = labels[labels == -1]
    if "nu" not in locals():
        nu = attacks.shape[0] / labels.shape[0]
        if nu == 0.0:
            nu = 0.001
    trainDataset.drop(["label"], axis=1, inplace=True)
    return nu, trainDataset

def prepare_test_dataset(testDataset):
    test_label = test_dataset.label
    testDataset.drop(["label"], axis=1, inplace=True)
    return test_label, testDataset

def describe_datasets(train_dataset,test_dataset):
    print("Zbior treningowy - ilosc probek: %s, ilosc atakow: %s, ilosc ruchu normalnego: %s" %(train_dataset.shape[0], train_dataset.loc[train_dataset["label"] != "normal."].shape[0], train_dataset.loc[train_dataset["label"] == "normal."].shape[0]))
    print("Zbior testowy - ilosc probek: %s, ilosc atakow: %s, ilosc ruchu normalnego: %s" % (test_dataset.shape[0], test_dataset.loc[test_dataset["label"] != "normal."].shape[0], test_dataset.loc[test_dataset["label"] == "normal."].shape[0]))

def VarianceThreshold_selector(data):
    columns = data.columns
    selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
    selector.fit_transform(data)
    labels = [columns[x] for x in selector.get_support(indices=True) if x]
    result_array = []
    for n in selector.get_support(indices=True):
        result_array = np.append(result_array, columns[n])
    result_array = np.append(result_array, 'label')
    print("Wybor istotnych parametrow: ",result_array)
    return result_array
def SelectKBest_selector(trainDataset, k):
    columns = trainDataset.columns
    y =trainDataset.label
    x = trainDataset
    #X_new = SelectKBest(chi2, k=20).fit_transform(x, y)
    xnew = SelectKBest(chi2, k=k)
    xnew.fit(x,y)
    idxs = xnew.get_support(indices=True)
    result_array=[]
    for n in idxs:
        result_array = np.append(result_array, columns[n])
    if "label" not in result_array:
        result_array = np.append(result_array, 'label')
    print("Wybor istotnych parametrow: ",result_array)
    return result_array


# Definicje
options = ['mode', 'kernel', "gamma","nu"]
kernels = ["rbf", "linear", "sigmoid", "poly"]
modes = ["all","tcp","udp","icmp"]
sysargv = pd.DataFrame(columns=options)
scan = False
nu=0.0001
# Inicjalizacja danych
if len(sys.argv) == 1 and sys.argv[0] not in modes:
    print("Przyklad uruchomienia: py oneclasssvm-ids.py all|tcp|udp|icmp <optcje> \ngdzie opcje:\nkernel=rbf|linear|sigmoid|poly (domyslnie rbf)\nnu=<0,1] (domyslnie obliczana na podstawie danych)\ngamma=skan wykonanie skanyu gamma - UWAGA: może trwać ponad godzinę")
    exit(1)
mode = sys.argv[1]
for arg in sys.argv:

    if "kernel" in arg and arg.split("=")[1] in kernels:
        kernel = arg.split("=")[1]
    else:
        kernel = "rbf"
    if "gamma=skan" == arg:
        scan=True
        print("UWAGA wybrano opcje skan, czas wykonania moze wyniesc ponad godzine")
    if "nu" in arg:
        if (float(arg.split("=")[1]) > 0 and float(arg.split("=")[1]) <= 1):
            nu = arg.split("=")[1]
        else:
            nu = 0.0001
if ~scan:
    print("Wybrane parametry:\n- tryb skanu : %s \n- kernel : %s \n- nu : %s\n- gamma : %s" % (mode, kernel, nu,0.55))
else:
    print("Wybrany tryb skanu parametru gamma dla :\n- tryb skanu : %s \n- kernel : %s \n- nu : %s\n- gamma : %s" % (mode, kernel, nu,0.55))

#Pobranie danych treningowych i testowych
column_names = ["duration","protocol_type","service","flag","zsrc_bytes","dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
train_dataset = pd.read_csv('kddcup.data_10_percent_corrected', names = column_names, index_col = False)
test_dataset = pd.read_csv('corrected', names = column_names, index_col = False)

if (mode != "all"):
    train_dataset = train_dataset.loc[train_dataset["protocol_type"] == mode]
    test_dataset = test_dataset.loc[test_dataset["protocol_type"] == mode]

if (test_dataset.shape[0] > 50000):
    test_dataset = test_dataset.sample(50000)
train_dataset = train_dataset.loc[train_dataset["label"] == "normal."]

describe_datasets(train_dataset,test_dataset)

train_dataset, test_dataset = convert_text_values(train_dataset, test_dataset)

features = VarianceThreshold_selector(train_dataset)
#features = column_names

#features = SelectKBest_selector(train_dataset,33)
train_dataset = train_dataset[features]
test_dataset = test_dataset[features]

# Przygotowanie danych treningowych oraz testowych
#train_dataset, test_dataset = convert_text_values(train_dataset, test_dataset)
nu, train_dataset = get_nu_and_strip_labels(train_dataset,nu)
test_preds, test_dataset = prepare_test_dataset(test_dataset)

train_dataset = normalize_features(train_dataset, features[:-1])
test_dataset = normalize_features(test_dataset, features[:-1])
start_time = time.time()

# Uruchomienie opcji znalezienia odpowiedniego gamma
if scan:
    nu = 0.1
    skok = 0.1
    columns = ['nu', 'dokladnosc', "precyzja", "falszywe ataki", "korelacja"]

    quality = pd.DataFrame(columns=columns)
    while nu < 1:
        model = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=0.55, cache_size=1024)
        model.fit(train_dataset)
        preds = model.predict(test_dataset)
        targs = test_preds
        TN, FP, FN, TP = metrics.confusion_matrix(targs, preds).ravel()
        prec = ((TP) / (float(TP) + float(FP)))
        dokl = ((TP + TN) / (TP + TN + FP + FN))
        fa = ((FP) / (FP + TN))
        corr = (((TN * TP) - (FN * FP)) / (math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))))
        quality.loc[len(quality)] = [nu, dokl, prec, fa, corr]
        print('go for ', nu)
        nu += skok
    plt.interactive(False)
    quality.plot(x='nu', marker='.')
    plt.show()
# Nauka modelu i weryfikacja wynikow
else:
    print("shape training", train_dataset.shape)
    print("shape test", test_dataset.shape)
    model = svm.OneClassSVM(nu=float(0.0001), kernel=kernel, gamma=0.55,cache_size=1024)
    model.fit(train_dataset)
    preds = model.predict(test_dataset)
    targs = test_preds
    print("Wyniki jakosci modelu dla: %s kernel=%s gamma=%s nu=%s coef0=%s degree=%s" % (mode, kernel, 0.9,nu, 0.0, 3))
    TN, FP, FN, TP = metrics.confusion_matrix(targs, preds).ravel()
    print("tn %s, tp %s, fn %s, fp %s" % (TN,TP,FN,FP))
    print("precision", ((TP)/(float(TP)+float(FP))))
    print("accuracy", ((TP+ TN)/(TP+TN+FP+FN)))
    print("false alarm", ((FP)/(FP+TN)))
    print("correlation", (((TN*TP)-(FN*FP))/(math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN)))))

print("--- %s seconds ---" % (time.time() - start_time))
