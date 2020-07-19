import torch
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def get_device(cuda=True):
    if cuda == True:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device('cpu')


def prepare_data_lstm(path):
    dataset = pd.read_csv(path)
    print('Raw Shape: ', dataset.shape)

    lag_columns = [c for c in dataset.columns if 'lag' in c]
    dataset = dataset[[c for c in dataset.columns if c not in lag_columns]]

    DateColumns = ['Date']

    ext_columns = ['Dow', 'arrival','maxtemp', 'mintemp', 'avgtemp', 'departure', 'hdd',
        'cdd', 'participation', 'newsnow', 'snowdepth', 'ifSnow']

    targetColumns = [c for c in dataset.columns if c not in ext_columns and \
                    c not in DateColumns and c not in lag_columns and c != 'Hour']

    features_cols = [c for c in dataset.columns if c not in targetColumns and c not in DateColumns]

    print('Cleaned Shape: ', dataset.shape)
    print('Target columns:', len(targetColumns))
    print("Feature coumns: ", len(features_cols))

    return dataset, targetColumns, features_cols
    

def create_data_sequences(x,y, bptt):
    """
    Takes ridership data and external features to 
        create sequences of data for training and
        testing.

    x: external features
    y: ridhership data
    bptt: sequence length
    """
    inout_seq = []
    for i in range(len(x)-bptt):
        train_seq_x = x[i:i+bptt]
        train_seq_y = y[i:i+bptt]

        train_label = y[i+1:i+bptt+1]
        inout_seq.append((train_seq_x, train_seq_y ,train_label))
    return inout_seq


def train_test_split_monthly(dataset, month):
    month_index  = pd.to_datetime(dataset.Date).dt.month == month
    trainData = dataset[~month_index]
    testData = dataset[month_index]
    print("train test split")
    print("train shape: ",trainData.shape)
    print("test shape: ",testData.shape)
    return trainData, testData


def prepare_data_tensors(trainData, testData, features_cols, targetColumns, device):
    X_train = trainData[features_cols].values
    X_train = torch.tensor(X_train).float().to(device)
    print("train feature tensor shape :",X_train.shape)

    y_train = trainData[targetColumns].values
    y_train = torch.tensor(y_train).float().to(device)
    print("train target tensor shape :",y_train.shape)

    X_test = testData[features_cols].values
    X_test = torch.tensor(X_test).float().to(device)
    print("test feature tensor shape :",X_test.shape)

    y_test = testData[targetColumns].values
    y_test = torch.tensor(y_test).float().to(device)    
    print("test target tensor shape :",y_test.shape)
    
    return X_train, y_train, X_test, y_test


def evaluate_lstm_pipeline_model(model, test_inout_seq, device):
    """
    Generate evaluation metrics for a given lstm pipline model
    """
    model.eval()
    prediction = []
    with torch.no_grad():
        for feat,seq, labels in test_inout_seq:
            model.initialize_hidden_cell(device)
            # = (torch.zeros(layers, 1, model.hidden_layer_size).to(device),
            #                 torch.zeros(layers, 1, model.hidden_layer_size).to(device))
            prediction.append(model(seq,feat)[-1])


    y_test_ = torch.stack([labels[-1] for feat,seq, labels in test_inout_seq], axis=0).detach().cpu().numpy()
    y_pred_ = torch.stack(prediction).detach().cpu().numpy()

    res = y_pred_ - y_test_
    r2 = r2_score(y_test_, y_pred_, multioutput='variance_weighted')
    rmse = mean_squared_error(y_test_, y_pred_)
    mae = mean_absolute_error(y_test_, y_pred_)
    return (res, r2, rmse, mae)


def get_community_attachment_matix(targetColumns, zone_to_comm_file):
    #zone_to_comm_file = '/home/urwa/Documents/side_projects/urban/UrbanTemporalNetworks/Data/ZonetoComm.csv'
    comms = pd.read_csv(zone_to_comm_file)  
    communities = list(set(comms.start_community))

    mapping = dict(zip(comms.start_id, comms.start_community))
    comm_to_index = dict(zip(communities,range(len(communities))))
    col_to_index = dict(zip(targetColumns,range(len(targetColumns))))

    attach = torch.zeros(len(targetColumns), len(communities))

    for t_c in targetColumns:
        com = mapping[int(t_c)]
        x_i = col_to_index[t_c]
        y_i = comm_to_index[com]

        attach[x_i,y_i] = 1

    return attach