import torch
import random
class Config:
    data_path = "./data"  # 数据集保存路径
    dataset_name = "Cora"  # 数据集名称，可选Cora, Citeseer, Pubmed
    save_path = './checkpoints/GCN/best_model.pth'  # 模型权重保存路径, 可选GCN，GAT......
    LEARNING_RATE = 0.01
    WEIGHT_DACAY = 5e-4
    EPOCHS = 400
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HIDDEN_DIM = 32
    NUM_LAYERS = 2
    PERCENT = 0.9
    seed = 42
    dropout = 0.6
    alpha1_intra = 0.2
    alpha2_inter = 0.02
    USE_SAMPLING = False
    WITH_BN = False
    Dynamic_Reorganization = True
    USE_SMO = True
    MODEL_NAME = 'DRS-GCN'
    accuracy_save_path = './acc/DenseGCN_ACC/Pubmed_no_track_64.csv'
