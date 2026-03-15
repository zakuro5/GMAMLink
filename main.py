import csv
import time
import psutil
import os

from utils import *
from GraphMAE import GraphMAE
from Linkmodel import LinkModel
from GCN import *
from torch.optim import Adam,SGD,Adamax
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


exp_file_array = [
    # STRING
    # 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hHEP/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hHEP/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mDC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mDC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mESC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-E/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-E/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-GM/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-GM/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-L/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-L/TFs+500/BL--ExpressionData.csv',
    # # Lofgof
    # 'Benchmark_Dataset/Lofgof_Dataset/mESC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Lofgof_Dataset/mESC/TFs+500/BL--ExpressionData.csv',
    # # Non-Specific
    # 'Benchmark_Dataset/Non-Specific_Dataset/hESC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hESC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hHEP/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hHEP/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mDC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mDC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mESC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mESC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-E/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-E/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-GM/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-GM/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-L/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-L/TFs+500/BL--ExpressionData.csv',
    # # Specific
    # 'Benchmark_Dataset/Specific_Dataset/hESC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hESC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hHEP/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hHEP/TFs+500/BL--ExpressionData.csv',
    'Benchmark_Dataset/Specific_Dataset/mDC/TFs+1000/BL--ExpressionData.csv',
    'Benchmark_Dataset/Specific_Dataset/mDC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mESC/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mESC/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-E/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-E/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-GM/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-GM/TFs+500/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-L/TFs+1000/BL--ExpressionData.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-L/TFs+500/BL--ExpressionData.csv'
]

tf_file_array = [
    # STRING
    # 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hHEP/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hHEP/TFs+500/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mDC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mDC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mESC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-E/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-E/TFs+500/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-GM/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-GM/TFs+500/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-L/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-L/TFs+500/TF.csv',
    # # Lofgof
    # 'Benchmark_Dataset/Lofgof_Dataset/mESC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Lofgof_Dataset/mESC/TFs+500/TF.csv',
    # # Non-Specific
    # 'Benchmark_Dataset/Non-Specific_Dataset/hESC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hESC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hHEP/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hHEP/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mDC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mDC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mESC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mESC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-E/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-E/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-GM/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-GM/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-L/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-L/TFs+500/TF.csv',
    # # Specific
    # 'Benchmark_Dataset/Specific_Dataset/hESC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hESC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hHEP/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hHEP/TFs+500/TF.csv',
    'Benchmark_Dataset/Specific_Dataset/mDC/TFs+1000/TF.csv',
    'Benchmark_Dataset/Specific_Dataset/mDC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mESC/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mESC/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-E/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-E/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-GM/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-GM/TFs+500/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-L/TFs+1000/TF.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-L/TFs+500/TF.csv'
]

target_file_array = [
    # STRING
    # 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hESC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hHEP/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/hHEP/TFs+500/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mDC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mDC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mESC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-E/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-E/TFs+500/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-GM/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-GM/TFs+500/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-L/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/STRING_Dataset/mHSC-L/TFs+500/Target.csv',
    # # Lofgof
    # 'Benchmark_Dataset/Lofgof_Dataset/mESC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Lofgof_Dataset/mESC/TFs+500/Target.csv',
    # # Non-Specific
    # 'Benchmark_Dataset/Non-Specific_Dataset/hESC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hESC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hHEP/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/hHEP/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mDC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mDC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mESC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mESC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-E/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-E/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-GM/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-GM/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-L/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Non-Specific_Dataset/mHSC-L/TFs+500/Target.csv',
    # # Specific
    # 'Benchmark_Dataset/Specific_Dataset/hESC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hESC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hHEP/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/hHEP/TFs+500/Target.csv',
    'Benchmark_Dataset/Specific_Dataset/mDC/TFs+1000/Target.csv',
    'Benchmark_Dataset/Specific_Dataset/mDC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mESC/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mESC/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-E/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-E/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-GM/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-GM/TFs+500/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-L/TFs+1000/Target.csv',
    # 'Benchmark_Dataset/Specific_Dataset/mHSC-L/TFs+500/Target.csv'
]

train_file_array = [
    # STRING
    # 'Benchmark_Dataset/STRING/hESC 1000/Train_set.csv',
    # 'Benchmark_Dataset/STRING/hESC 500/Train_set.csv',
    # 'Benchmark_Dataset/STRING/hHEP 1000/Train_set.csv',
    # 'Benchmark_Dataset/STRING/hHEP 500/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mDC 1000/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mDC 500/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mESC 1000/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-E 1000/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-E 500/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-GM 1000/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-GM 500/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-L 1000/Train_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-L 500/Train_set.csv',
    # # Lofgof
    # 'Benchmark_Dataset/Lofgof/mESC 1000/Train_set.csv',
    # 'Benchmark_Dataset/Lofgof/mESC 500/Train_set.csv',
    # # Non-Specific
    # 'Benchmark_Dataset/Non-Specific/hESC 1000/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hESC 500/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hHEP 1000/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hHEP 500/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mDC 1000/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mDC 500/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mESC 1000/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mESC 500/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-E 1000/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-E 500/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-GM 1000/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-GM 500/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-L 1000/Train_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-L 500/Train_set.csv',
    # # Specific
    # 'Benchmark_Dataset/Specific/hESC 1000/Train_set.csv',
    # 'Benchmark_Dataset/Specific/hESC 500/Train_set.csv',
    # 'Benchmark_Dataset/Specific/hHEP 1000/Train_set.csv',
    # 'Benchmark_Dataset/Specific/hHEP 500/Train_set.csv',
    'Benchmark_Dataset/Specific/mDC 1000/Train_set.csv',
    'Benchmark_Dataset/Specific/mDC 500/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mESC 1000/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mESC 500/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-E 1000/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-E 500/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-GM 1000/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-GM 500/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-L 1000/Train_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-L 500/Train_set.csv'
]

val_file_array = [
    # STRING
    # 'Benchmark_Dataset/STRING/hESC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/hESC 500/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/hHEP 1000/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/hHEP 500/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mDC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mDC 500/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mESC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-E 1000/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-E 500/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-GM 1000/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-GM 500/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-L 1000/Validation_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-L 500/Validation_set.csv',
    # # Lofgof
    # 'Benchmark_Dataset/Lofgof/mESC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Lofgof/mESC 500/Validation_set.csv',
    # # Non-Specific
    # 'Benchmark_Dataset/Non-Specific/hESC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hESC 500/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hHEP 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hHEP 500/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mDC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mDC 500/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mESC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mESC 500/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-E 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-E 500/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-GM 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-GM 500/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-L 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-L 500/Validation_set.csv',
    # # Specific
    # 'Benchmark_Dataset/Specific/hESC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/hESC 500/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/hHEP 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/hHEP 500/Validation_set.csv',
    'Benchmark_Dataset/Specific/mDC 1000/Validation_set.csv',
    'Benchmark_Dataset/Specific/mDC 500/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mESC 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mESC 500/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-E 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-E 500/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-GM 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-GM 500/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-L 1000/Validation_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-L 500/Validation_set.csv'
]

test_file_array = [
    # STRING
    # 'Benchmark_Dataset/STRING/hESC 1000/Test_set.csv',
    # 'Benchmark_Dataset/STRING/hESC 500/Test_set.csv',
    # 'Benchmark_Dataset/STRING/hHEP 1000/Test_set.csv',
    # 'Benchmark_Dataset/STRING/hHEP 500/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mDC 1000/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mDC 500/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mESC 1000/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-E 1000/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-E 500/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-GM 1000/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-GM 500/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-L 1000/Test_set.csv',
    # 'Benchmark_Dataset/STRING/mHSC-L 500/Test_set.csv',
    # # Lofgof
    # 'Benchmark_Dataset/Lofgof/mESC 1000/Test_set.csv',
    # 'Benchmark_Dataset/Lofgof/mESC 500/Test_set.csv',
    # # Non-Specific
    # 'Benchmark_Dataset/Non-Specific/hESC 1000/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hESC 500/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hHEP 1000/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/hHEP 500/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mDC 1000/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mDC 500/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mESC 1000/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mESC 500/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-E 1000/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-E 500/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-GM 1000/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-GM 500/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-L 1000/Test_set.csv',
    # 'Benchmark_Dataset/Non-Specific/mHSC-L 500/Test_set.csv',
    # # Specific
    # 'Benchmark_Dataset/Specific/hESC 1000/Test_set.csv',
    # 'Benchmark_Dataset/Specific/hESC 500/Test_set.csv',
    # 'Benchmark_Dataset/Specific/hHEP 1000/Test_set.csv',
    # 'Benchmark_Dataset/Specific/hHEP 500/Test_set.csv',
    'Benchmark_Dataset/Specific/mDC 1000/Test_set.csv',
    'Benchmark_Dataset/Specific/mDC 500/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mESC 1000/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mESC 500/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-E 1000/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-E 500/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-GM 1000/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-GM 500/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-L 1000/Test_set.csv',
    # 'Benchmark_Dataset/Specific/mHSC-L 500/Test_set.csv'
]

result_array = []

def train(exp_file, tf_file, target_file, train_file, val_file, test_file):
    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
    target = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)
    feature = torch.from_numpy(feature)
    tf = torch.from_numpy(tf)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # device = 'cpu'

    data_feature = feature.to(device)
    tf = tf.to(device)

    try:
        train_data = pd.read_csv(train_file, index_col=0).values
        validation_data = pd.read_csv(val_file, index_col=0).values
        test_data = pd.read_csv(test_file, index_col=0).values
    except FileNotFoundError as e:
        print(f"Skipping experiment, file not found: {e.filename}")
        return
    except Exception as e:
        print(f"Skipping experiment, error reading split files: {e}")
        return

    train_load = scRNADataset(train_data, feature.shape[0], flag=False)
    adj = train_load.Adj_Generate(tf, loop=False)

    adj = adj2saprse_tensor(adj)
    adj = adj.to_dense()

    train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(validation_data)
    test_data = torch.from_numpy(test_data)

    test_data = test_data.to(device)
    train_data = train_data.to(device)
    validation_data = val_data.to(device)

    model = GraphMAE(input_dim=feature.size()[1],
                     num_hidden=256,
                     num_layers=2,
                     output_dim=16,
                     device=device,
                     )

    # GraphMAE train

    model = model.to(device)
    optimizer = Adamax(model.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    adj = adj.to(device)
    model = model.to(device)

    epochs = 200
    batch_size = 256
    # logging.info('training..')
    epoch_iter = tqdm(range(epochs))
    
    # 添加早停机制
    best_loss = float('inf')
    early_stop_patience = 20
    early_stop_counter = 0
    
    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(data_feature, adj)
        
        # 早停机制
        if loss < best_loss:
            best_loss = loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_graphmae.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()

    a = model.encode(data_feature, adj)
    print(data_feature.shape[1])
    linkmodel = LinkModel(input_dim=data_feature.shape[1], origin_output_dim=256, hidden_dim=128, output_dim=32, pretrain_dim=256)
    linkmodel = linkmodel.to(device)
    optimizer = Adam(linkmodel.parameters(), lr=5e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    AUC, AUPR, AUPR_norm = 0, 0, 0
    best_AUC, best_AUPR, best_AUPR_norm = 0, 0, 0
    
    epochs = 10
    batch_size = 128
    for epoch in range(epochs):
        linkmodel.train()
        running_loss = 0.0
        for train_x, train_y in DataLoader(train_load, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            train_x = train_x.to(device)
            train_y = train_y.to(device).view(-1, 1)

            pred = linkmodel(a.data, train_x, data_feature, adj, None)
            pred = torch.sigmoid(pred)

            loss_BCE = F.mse_loss(pred, train_y)

            loss_BCE.backward()

            torch.nn.utils.clip_grad_norm_(linkmodel.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss_BCE.item()

        scheduler.step()
        linkmodel.eval()
        score = linkmodel(a.data, test_data, data_feature, adj, None)

        score = torch.sigmoid(score)

        score = torch.sigmoid(score)
        AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=test_data[:, -1], flag=False)
        print('Epoch:{}'.format(epoch + 1),
              'train loss:{}'.format(running_loss),
              'AUC:{:.3F}'.format(AUC),
              'AUPR:{:.3F}'.format(AUPR))
        if AUC > best_AUC:
            best_AUC = AUC
            best_AUPR = AUPR
            best_AUPR_norm = AUPR_norm
            # torch.save(linkmodel.state_dict(), 'best_model.pth')
            
        #
        # print('Epoch:{}'.format(epoch + 1),
        #       'train loss:{}'.format(running_loss),
        #       'AUC:{:.3F}'.format(AUC),
        #       'AUPR:{:.3F}'.format(AUPR))
    return best_AUC, best_AUPR, best_AUPR_norm

# TODO: 消融 去掉GraphMAE 直接下游推断； 去掉Attention和原始数据
if __name__ == '__main__':
    result_array = []
    for i in range(len(exp_file_array)):
        print('----------------------------------------------------------------------------------------------------------')
        print('Running experiment {}/{}'.format(i + 1, len(exp_file_array)))
        print('Expression File: {}'.format(exp_file_array[i]))

        # Start timing and memory tracking
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 # MB

        result = train(exp_file_array[i], tf_file_array[i], target_file_array[i], train_file_array[i], val_file_array[i], test_file_array[i])
        
        # End timing and memory tracking
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        elapsed_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        gpu_memory_usage = 0
        if torch.cuda.is_available():
             peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 # MB
             gpu_memory_usage = peak_gpu_memory

        if result is None:
            continue
            
        AUC,AUPR,AUPR_norm = result
        print(
              'AUC:{:.3F}'.format(AUC),
              'AUPR:{:.3F}'.format(AUPR),
              'Time:{:.2f}s'.format(elapsed_time),
              'RAM Usage:{:.2f}MB'.format(memory_usage),
              'Peak GPU Memory:{:.2f}MB'.format(gpu_memory_usage))
              
        result_array.append((exp_file_array[i], AUC, AUPR, elapsed_time, memory_usage, gpu_memory_usage))
