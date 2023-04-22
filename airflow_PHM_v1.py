import pandas as pd
import numpy as np

import os
import random
from collections import Counter
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from scipy import interpolate

import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import wandb

import pendulum
import airflow
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

####################################################################################################

# 하이퍼파라미터
class cfg:
    seed=1234
    gpu_idx = 0
    device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
    
    input_dim = 512
    output_dim = 8
    
    batch_size=32
    epochs = 300
    learning_rate=0.0001
    
    check_epoch = 1
    dropout = 0.05
    reg_lambda = 0
    
    # 경로설정
    base_path = "/home/seon/workspace/competition/aifactory/anomaly_detection_comp"
    
    model_name = 'AutoEncoder_512_8'

# 시드 고정 
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
seed_everything(cfg.seed)

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoder, self).__init__()
        self.input_dim = cfg.input_dim
        self.dropout = cfg.dropout
        self.basic_AE()
    
    # AutoEncoder - 기본형
    def basic_AE(self):
        self.encoding_layer = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.input_dim//2, bias=False),
            nn.Linear(cfg.input_dim//2, cfg.input_dim//4, bias=False),
            nn.Linear(cfg.input_dim//4, cfg.input_dim//8, bias=False),
            nn.Linear(cfg.input_dim//8, cfg.input_dim//16, bias=False),
            nn.Linear(cfg.input_dim//16, cfg.input_dim//32, bias=False),
            nn.Linear(cfg.input_dim//32, cfg.input_dim//64, bias=False),
            nn.Linear(cfg.input_dim//64, cfg.output_dim, bias=True)
        )

        self.decoding_layer = nn.Sequential(
            nn.Linear(cfg.output_dim, cfg.input_dim//64, bias=True),
            nn.Linear(cfg.input_dim//64, cfg.input_dim//32, bias=False),
            nn.Linear(cfg.input_dim//32, cfg.input_dim//16, bias=False),
            nn.Linear(cfg.input_dim//16, cfg.input_dim//8, bias=False),
            nn.Linear(cfg.input_dim//8, cfg.input_dim//4, bias=False),
            nn.Linear(cfg.input_dim//4, cfg.input_dim//2, bias=False),
            nn.Linear(cfg.input_dim//2, cfg.input_dim, bias=False)
        )
        self.apply(self._init_weights)
    
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, x):
        latent = self.encoding_layer(x)
        recon_input = self.decoding_layer(latent)
        return recon_input


# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, data):
        self.features = data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features.iloc[idx])

####################################################################################################
# Preprocessing
def _preprocessing(scaler, hptype, input_num):
    # 데이터 로드
    train_data = pd.read_csv(os.path.join(cfg.base_path, 'dataset', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(cfg.base_path, 'dataset', 'test_data.csv'))

    # motor_vibe 이상치 제거
    outlier_idx = train_data[train_data.motor_vibe > 6].index
    train_data.drop(index=outlier_idx, inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    
    # type을 마력별로 대체
    def type2hp(x):
        if x == 1:
            return 20
        elif x == 2:
            return 10
        elif x == 3:
            return 50
        else:
            return 30
        
    train_data.type = train_data.type.apply(lambda x: type2hp(x))
    test_data.type = test_data.type.apply(lambda x: type2hp(x))
    
    # 컬럼순서 변경
    new_column = ['out_pressure', 'air_inflow', 'air_end_temp', 'motor_current',
       'motor_rpm', 'motor_temp', 'motor_vibe', 'type']
    train_data = train_data[new_column]
    test_data = test_data[new_column]

    # 특정 타입의 데이터프레임만 학
    train_data = train_data[train_data.type == hptype]
    test_data = test_data[test_data.type == hptype]
    
    # scaling
    scale_columns = ['air_inflow', 'air_end_temp', 'motor_current', 
                     'motor_rpm', 'motor_temp', 'motor_vibe']
    scaled_train = pd.DataFrame(scaler.fit_transform(train_data[scale_columns]), 
                                index=train_data.index, columns=scale_columns)
    scaled_test = pd.DataFrame(scaler.transform(test_data[scale_columns]),
                               index=test_data.index, columns=scale_columns)

    # scaled_train["type"] = train_data.type
    # scaled_test["type"] = test_data.type

    # 미분
    deriv_train = pd.DataFrame()
    deriv_test = pd.DataFrame()
    deriv_train["out_pressure"] = train_data.out_pressure
    deriv_test["out_pressure"] = test_data.out_pressure
    for i in range(5):
        diff_train = scaled_train.apply(lambda x: x[i+1] - x[i], axis=1)
        deriv_train[f'diff{i+1}'] = diff_train
        diff_test = scaled_test.apply(lambda x: x[i+1] - x[i], axis=1)
        deriv_test[f'diff{i+1}'] = diff_test
    

    x = np.arange(0, len(deriv_train.columns))
    y1 = deriv_train.values
    y2 = deriv_test.values
    
    xnew = np.linspace(0, len(deriv_train.columns)-1, input_num)
    
    # 오버샘플링
    ynew = [interpolate.interp1d(x, y1[i], kind='linear')(xnew) for i in range(len(deriv_train))]
    input_train = pd.DataFrame(ynew, index=deriv_train.index)
    #input_train["type"] = hptype
    
    ynew = [interpolate.interp1d(x, y2[i], kind='linear')(xnew) for i in range(len(deriv_test))]
    input_test = pd.DataFrame(ynew, index=deriv_test.index)
    #input_test["type"] = hptype
    
    # csv형태로 저장. pickle은 보안 위험, 데이터프레임이 복잡하지 않으므로, json이 아닌 csv 이용
    input_train.to_csv(os.path.join(cfg.base_path,'tempDir',f'train_{hptype}.csv'))
    input_test.to_csv(os.path.join(cfg.base_path,'tempDir',f'test_{hptype}.csv'))
    
####################################################################################################
# 모델 학습
def _modeling(modelName, hptype):
    # WandB conf.
    wandb.init(
    project = "AirCompressor",
    entity = "gkrtjs0122",
    config = {
        "learning_rate": cfg.learning_rate,
        "architecture": modelName,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
    })
    
    # trainset
    input_train = pd.read_csv(os.path.join(cfg.base_path, 'tempDir',f'train_{hptype}.csv'), index_col=0)
    train_dataset=CustomDataset(input_train)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    
    # testset
    input_test = pd.read_csv(os.path.join(cfg.base_path, 'tempDir',f'test_{hptype}.csv'), index_col=0)
    test_dataset=CustomDataset(input_test)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)
    
    # model    
    model = AutoEncoder(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)
    criterion = torch.nn.L1Loss()
    
    def calc_avg(avg, val, idx):
        return (avg * idx + val) / (idx + 1)
    
    # Early-Stopping conf.
    best_loss = 10**4
    patience_limit = 100
    patience_cnt = 0
    
    # training
    for epoch in range(cfg.epochs+1):
        model.train()
        running_loss_avg = 0.0
        
        for idx, input in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            input = input.to(cfg.device)
            output = model(input)
            loss = criterion(output, input)
    
            loss.backward()
            optimizer.step()
            
            if torch.isnan(loss):
                print('Loss NaN. Train Finish.')
                break
            
            running_loss_avg = calc_avg(running_loss_avg, loss, idx)
            
        result = np.around(running_loss_avg.item(), 5)
        if result > best_loss:
            patience_cnt += 1
            if patience_cnt >= patience_limit:
                print('Early Stopping ... ')
                break
        else:
            best_loss = result
            patience_cnt = 0
        wandb.log({f'Loss_{hptype}HP': result})
        print('[%d, %5d] loss: %.5f' % (epoch+1, idx+1, running_loss_avg.item()))
        
    
    print('Finished Training')
    torch.save(model.state_dict(), os.path.join(cfg.base_path,'model',f'{modelName}_{hptype}.pth'))
    wandb.save(os.path.join(cfg.base_path,'model',f'{modelName}_{hptype}.pth'))
    
    # Scoring
    train_score = list()
    model.eval()
    with torch.no_grad():
        for input in train_dataloader:
            input.to(cfg.device)
            output = model(input)
            score = torch.mean(torch.abs(output - input), axis=1)
            train_score.extend(score.cpu().numpy())
    threshold = np.array(train_score).max()
    
    test_score = list()
    with torch.no_grad():
        for input in test_dataloader:
            input.to(cfg.device)
            output = model(input)
            score = torch.mean(torch.abs(output - input), axis=1)
            test_score.extend(score.cpu().numpy())
    
    test_score = np.array(test_score)
    test_score = np.where(test_score <= threshold, 0, test_score)
    test_score = np.where(test_score > threshold, 1, test_score)
    cnt = Counter(test_score)
    wandb.log({f'Anomly_{hptype}HP': cnt[1]})
    
    
    idx = input_test.index
    temp_submission = pd.DataFrame(test_score, index = input_test.index, columns=["label"])
    temp_submission.to_csv(os.path.join(cfg.base_path, 'tempDir', f'df{hptype}.csv'))

def _join_results(modelName):
    df10 = pd.read_csv(os.path.join(cfg.base_path, 'tempDir', 'df10.csv'), index_col=0)
    df20 = pd.read_csv(os.path.join(cfg.base_path, 'tempDir', 'df20.csv'), index_col=0)
    df30 = pd.read_csv(os.path.join(cfg.base_path, 'tempDir', 'df30.csv'), index_col=0)
    df50 = pd.read_csv(os.path.join(cfg.base_path, 'tempDir', 'df50.csv'), index_col=0)
    
    submission = pd.read_csv(os.path.join(cfg.base_path, 'answer_sample.csv')) 
    for df in [df10, df20, df30, df50]:
        idx = df.index
        submission.loc[idx, 'label'] = df.label.to_numpy()
    
    now = pendulum.now()
    timeInfo = now.strftime('%m%d%H%M')
    submission.to_csv(os.path.join(cfg.base_path, 'submission',
                                   f'{modelName}_{timeInfo}.csv'), index=False)
    

####################################################################################################
    
dag = DAG(
    dag_id="air_compressor_PHM",
    start_date=pendulum.today('UTC').add(days=-1),
    schedule='@once',
)

start = EmptyOperator(task_id='start', dag=dag)

# TYPE_0,4,5,6,7 ######################################
data_preprocessing_30hp = PythonOperator(
    task_id = 'data_preprocessing_30hp',
    python_callable = _preprocessing,
    op_kwargs = {'scaler': StandardScaler(),
                 'hptype': 30,
                 'input_num': cfg.input_dim},
    dag = dag,
)

modeling_30hp = PythonOperator(
    task_id = 'modeling_30hp',
    python_callable = _modeling,
    op_kwargs = {'modelName': cfg.model_name,
                 'hptype': 30},
    dag = dag,
)
# TYPE_1 ##############################################
data_preprocessing_20hp = PythonOperator(
    task_id = 'data_preprocessing_20hp',
    python_callable = _preprocessing,
    op_kwargs = {'scaler': StandardScaler(),
                 'hptype': 20,
                 'input_num': cfg.input_dim},
    dag = dag,
)

modeling_20hp = PythonOperator(
    task_id = 'modeling_20hp',
    python_callable = _modeling,
    op_kwargs = {'modelName': cfg.model_name,
                 'hptype': 20},
    dag = dag,
)
# TYPE_2 ##############################################
data_preprocessing_10hp = PythonOperator(
    task_id = 'data_preprocessing_10hp',
    python_callable = _preprocessing,
    op_kwargs = {'scaler': StandardScaler(),
                 'hptype': 10,
                 'input_num': cfg.input_dim},
    dag = dag,
)

modeling_10hp = PythonOperator(
    task_id = 'modeling_10hp',
    python_callable = _modeling,
    op_kwargs = {'modelName': cfg.model_name,
                 'hptype': 10},
    dag = dag,
)
# TYPE_3 ##############################################
data_preprocessing_50hp = PythonOperator(
    task_id = 'data_preprocessing_50hp',
    python_callable = _preprocessing,
    op_kwargs = {'scaler': StandardScaler(),
                 'hptype': 50,
                 'input_num': cfg.input_dim},
    dag = dag,
)

modeling_50hp = PythonOperator(
    task_id = 'modeling_50hp',
    python_callable = _modeling,
    op_kwargs = {'modelName': cfg.model_name,
                 'hptype': 50},
    dag = dag,
)
#######################################################

join_results = PythonOperator(
    task_id = 'join_results',
    python_callable = _join_results,
    op_kwargs = {'modelName': cfg.model_name},
    dag = dag,
)

end = EmptyOperator(task_id='end', dag=dag)

####################################################################################################

start >> [data_preprocessing_30hp, data_preprocessing_20hp,
          data_preprocessing_10hp, data_preprocessing_50hp]
data_preprocessing_30hp >> modeling_30hp
data_preprocessing_20hp >> modeling_20hp
data_preprocessing_10hp >> modeling_10hp
data_preprocessing_50hp >> modeling_50hp
[modeling_30hp, modeling_20hp,
 modeling_10hp, modeling_50hp] >> join_results >> end