import torch
import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random

def get_device(cuda=True):
    """
    Get device to do the computation. If cuda is true it tries to get a gpu if avaialble.
    """
    if cuda == True:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        return torch.device('cpu')


def prepare_data_lstm(path):
    """
    Prepares the dataframe for lstm models. 
    
    Arguments:
    path: path of csv file which contains hourly level
            traffic data and external features. 

    Processing:
    - removes lagged variables if present since lstm does not need them
    - extract target columns: columns containing ridership data
    - extract feature columns: columns containing external features and hour

    Retruns:
    - cleaned dataset: (dataframe)
    - target columns: (list)
    - feature columns: (list)

    """
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

    Arguments:
    x: external features tensor
    y: ridhership data tensor
    bptt: sequence length for lstm

    Returns:
    inout_seq: list of training data and labels

    """
    inout_seq = []
    for i in range(len(x)-bptt):
        train_seq_x = x[i:i+bptt]
        train_seq_y = y[i:i+bptt]

        train_label = y[i+1:i+bptt+1]
        inout_seq.append((train_seq_x, train_seq_y ,train_label))
    return inout_seq


def train_test_split_monthly(dataset, month):
    """
    Divides the dataset dataframe into train and test dataframes
    based on given month
    """
    month_index  = pd.to_datetime(dataset.Date).dt.month == month
    trainData = dataset[~month_index]
    testData = dataset[month_index]
    print("train test split")
    print("train shape: ",trainData.shape)
    print("test shape: ",testData.shape)
    return trainData, testData

def train_test_split_temporal(dataset, train_ratio=0.75):
    """
    Divides the dataset dataframe into train and test dataframes
    temporally based on given training set ratio
    """
    sep = int(train_ratio*len(dataset))

    trainData = dataset[:sep]
    testData = dataset[sep:]

    print("train test split")
    print("train shape: ",trainData.shape)
    print("test shape: ",testData.shape)
    return trainData, testData

def prepare_data_tensors(trainData, testData, features_cols, targetColumns, device):
    """
    Divides the training and testing data into feature and target tensors
    Feature tensors contain external features
    Target tensors contian ridership data.
    """
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

    if torch.__version__.startswith('0.4'):
        y_test_ = torch.stack([labels[-1] for feat,seq, labels in test_inout_seq], dim=0).detach().cpu().numpy()
    else:
        y_test_ = torch.stack([labels[-1] for feat,seq, labels in test_inout_seq], axis=0).detach().cpu().numpy()

    
    y_pred_ = run_inference(model, test_inout_seq, device)

    res = y_pred_ - y_test_
    r2 = r2_score(y_test_, y_pred_, multioutput='variance_weighted')
    rmse = mean_squared_error(y_test_, y_pred_)
    mae = mean_absolute_error(y_test_, y_pred_)
    return (res, r2, rmse, mae)


def run_inference(model, test_inout_seq, device):
    """
    Runs inference on given data sequences and returns results
    """
    model.eval()
    prediction = []
    with torch.no_grad():
        for feat,seq, labels in test_inout_seq:
            model.initialize_hidden_cell(device)
            prediction.append(model(seq,feat)[-1])


    y_pred_ = torch.stack(prediction).detach().cpu().numpy()

    return y_pred_


def get_community_attachment_matix(targetColumns, zone_to_comm_file):
    """
    Reads taxi zone to community mapping from a file and creates an 
    attachment matrix
    """
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



def store_chekpoint(exp_dir, model,optimizer,lr_scheduler):
    """
    Saves the state of model, optimizer and scheduler 
    in experiment directory.
    """
    print('------- Saving checkpoint---------------')
    checkpoint_path = os.path.join(exp_dir,'checkpoint.pth')

    torch.save({'model_state_dict': model.state_dict(),                                                 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict()}, 
               checkpoint_path)



def load_chekpoint(exp_dir, model,optimizer,lr_scheduler):
    """
    Loads the state of model, optimizer and scheduler 
    in experiment directory.
    """
    print('------- Loading checkpoint---------------')
    checkpoint_path = os.path.join(exp_dir,'checkpoint.pth')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return model, optimizer, lr_scheduler


def lstm_monthly_dataloader(dataset, features_cols, targetColumns, month, bptt, device, shuffle=False):

    """
    Creates monthly training and testing data from dataset
    """
    
    trainData, testData = train_test_split_monthly(dataset, month)

    X_train, y_train, X_test, y_test = prepare_data_tensors(trainData, testData, 
                                                            features_cols, targetColumns, device)


    train_inout_seq = create_data_sequences(X_train,y_train, bptt)
    test_inout_seq = create_data_sequences(X_test,y_test, bptt)
    print("\nsequences")
    print(train_inout_seq[0][0].shape,train_inout_seq[0][1].shape, train_inout_seq[0][2].shape)
        
    if shuffle:
        random.shuffle(train_inout_seq)

    return train_inout_seq, test_inout_seq


def lstm_temporal_dataloader(dataset, features_cols, targetColumns, train_ratio, bptt, device, shuffle=False):
    
    """
    Creates training and testing datasets from dataset
    """

    trainData, testData = train_test_split_temporal(dataset, train_ratio)

    X_train, y_train, X_test, y_test = prepare_data_tensors(trainData, testData, 
                                                            features_cols, targetColumns, device)


    train_inout_seq = create_data_sequences(X_train,y_train, bptt)
    test_inout_seq = create_data_sequences(X_test,y_test, bptt)
    print("\nsequences")
    print(train_inout_seq[0][0].shape,train_inout_seq[0][1].shape, train_inout_seq[0][2].shape)
    
    if shuffle:
        random.shuffle(train_inout_seq)

    return train_inout_seq, test_inout_seq



def train_one_epoch(model, optimizer, loss_function, train_inout_seq, device):
    """
    Trains one epoch on data and returns the avg training loss
    """

    model.train()
    random.shuffle(train_inout_seq)
    
    losses = []
    for feat,seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.initialize_hidden_cell(device)
        y_pred = model(seq, feat)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

        losses.append(single_loss.item())
    return np.mean(losses)   
        


def get_edge_pred_df(y_pred_, targetColumns, weights_path):
    """
    Creates taxi zone level prediction from aggregated prediction
    based on the aggregation to zone weights file
    """

    edge_prediction_df = pd.DataFrame(y_pred_)
    edge_prediction_df.columns = targetColumns 

    zone_weights = pd.read_csv(weights_path)
    zone_weights['Borough'] = zone_weights['Borough'].astype(str)

    boroughs = list(edge_prediction_df.columns)
    for bor in boroughs:

        weight_df = zone_weights[zone_weights.Borough == bor]

        for b_zone,z_weight in zip(weight_df.DOLocationID.values,weight_df.zone_weight.values):        
            edge_prediction_df[b_zone] = edge_prediction_df[bor] * z_weight

    select_cols = [c for c in edge_prediction_df.columns if c not in boroughs]
    edge_prediction_df = edge_prediction_df[select_cols]

    return edge_prediction_df


def load_edge_test_data_temporal(file_path, train_ratio, bptt):
    """
    Loads the test taxi zone level data from given file and 
    preprocesses it for temporal test split.
    """
    edge_testData = pd.read_csv(file_path)
    
    sep = int(train_ratio*len(edge_testData))
    DateColumns = ['Date']

    ext_columns = ['Dow', 'arrival','maxtemp', 'mintemp', 'avgtemp', 'departure', 'hdd',
           'cdd', 'participation', 'newsnow', 'snowdepth', 'ifSnow']
    
    edge_testData = edge_testData[sep+bptt:]

    lag_columns = [c for c in edge_testData.columns if 'lag' in c]
    edge_testData = edge_testData[[c for c in edge_testData.columns if c not in lag_columns]]

    edgeColumns = [c for c in edge_testData.columns if c not in ext_columns and \
                    c not in DateColumns and c not in lag_columns and c != 'Hour']

    edge_testData = edge_testData[edgeColumns]

    return edge_testData

def load_edge_test_data_monthly(file_path, month, bptt):
    """
    Loads the test taxi zone level data from given file and 
    preprocesses it for monthly test split.
    """
    edge_testData = pd.read_csv(file_path)

    month_index  = pd.to_datetime(edge_testData.Date).dt.month == month
    
    #sep = int(train_ratio*len(edge_testData))
    DateColumns = ['Date']

    ext_columns = ['Dow', 'arrival','maxtemp', 'mintemp', 'avgtemp', 'departure', 'hdd',
           'cdd', 'participation', 'newsnow', 'snowdepth', 'ifSnow']
    
    edge_testData = edge_testData[month_index]
    edge_testData = edge_testData[bptt:]

    lag_columns = [c for c in edge_testData.columns if 'lag' in c]
    edge_testData = edge_testData[[c for c in edge_testData.columns if c not in lag_columns]]

    edgeColumns = [c for c in edge_testData.columns if c not in ext_columns and \
                    c not in DateColumns and c not in lag_columns and c != 'Hour']

    edge_testData = edge_testData[edgeColumns]

    return edge_testData

def evaluate_edge_temporal(model, test_inout_seq, device, targetColumns, weights_path, test_data_path, train_ratio, bptt):
    """
    Generate taxi zone level evaluation metrics for a given lstm model
    with temporal train test split.
    """
    y_pred_ = run_inference(model, test_inout_seq, device)
    edge_prediction_df = get_edge_pred_df(y_pred_, targetColumns, weights_path)
    edge_testData = load_edge_test_data_temporal(test_data_path, train_ratio, bptt)
    res, r2, rmse, mae = edge_metrics(edge_testData, edge_prediction_df)
    return res, r2, rmse, mae

def evaluate_edge_monthy(model, test_inout_seq, device, targetColumns, weights_path, test_data_path, month, bptt):
    """
    Generate taxi zone level evaluation metrics for a given lstm model
    with monthly train test split.
    """
    y_pred_ = run_inference(model, test_inout_seq, device)
    edge_prediction_df = get_edge_pred_df(y_pred_, targetColumns, weights_path)
    edge_testData = load_edge_test_data_monthly(test_data_path, month, bptt)
    res, r2, rmse, mae = edge_metrics(edge_testData, edge_prediction_df)
    return res, r2, rmse, mae

def edge_metrics(testdf, preddf):
    """
    Syncs the columns in two dataframes and compute metrics.
    """
    preddf.columns = preddf.columns.astype(str)
    testdf.columns = testdf.columns.astype(str)
    preddf = preddf[testdf.columns]
    
    y_pred_ = preddf.values
    y_test_ = testdf.values
    
    res = y_pred_ - y_test_
    r2 = r2_score(y_test_, y_pred_, multioutput='variance_weighted')
    rmse = mean_squared_error(y_test_, y_pred_)
    mae = mean_absolute_error(y_test_, y_pred_)
    return (res, r2, rmse, mae)