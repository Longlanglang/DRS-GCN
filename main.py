import os

import torch, random
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from model import *
from utils import *
import csv
from config import Config
from edge_curvature import GetEdgeCurvature
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = Config()
    get_random_seed(args.seed)
    # 数据准备
    dataset = Planetoid(root=args.data_path, name=args.dataset_name, transform=NormalizeFeatures())
    data = dataset[0]
    data = data.to(args.DEVICE)
    if args.Dynamic_Reorganization:
        get_curv = GetEdgeCurvature(data)
        ricci_curvature = get_curv.compute_ricci_curvature()
    else:
        # without dynamic
        ricci_curvature = torch.zeros(data.edge_index.shape[1]).to(args.DEVICE)

    input_dim = data.num_features
    output_dim = dataset.num_classes
    nodes_not_in_split = ~data.train_mask & ~data.val_mask & ~data.test_mask
    data.train_mask[nodes_not_in_split] = True
    # train/val/test:1208/500/1000


    def tensor_from_numpy(x, device):
        return torch.from_numpy(x).to(device)


    # 模型建立
    model_dict = {
        'GCN': GCN,
        'DRS-GCN': GCN2,
        'GAT': GAT,
        'ResGCN': ResGCN,
        'JKNet': JKNet,
        'IncepGCN': IncepGCN,
        'GraphSAGE': GraphSAGE,
        'DenseGCN': DenseGCN
    }

    if args.MODEL_NAME in model_dict:
        model_class = model_dict[args.MODEL_NAME]
        model = model_class(input_dim, args.HIDDEN_DIM, output_dim, args.NUM_LAYERS, args.seed, args.dropout,
                            args.USE_SAMPLING, args.Dynamic_Reorganization, args.WITH_BN).to(args.DEVICE)
    else:
        raise NotImplementedError("The model is not defined!")

    best_val_acc = 0.0  # 跟踪最佳验证集准确率
    early_stop_count = 0  # 用于早停的计数器
    patience = 50  # 早停的阈值

    criterion = torch.nn.CrossEntropyLoss().to(args.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DACAY)

    # 迭代训练
    for epoch in range(1, args.EPOCHS + 1):
        loss, train_accuracy, val_accuracy = train(model, data, optimizer, criterion, args.PERCENT,
                                                   ricci_curvature, args.USE_SMO, args.alpha1_intra, args.alpha2_inter)
        if epoch % 10 == 0:
            print('Epoch:{:03d} , loss:{:.4f} , Train_Accuracy:{:.4f} , Val_Accuracy:{:.4f}'
                  .format(epoch, loss, train_accuracy, val_accuracy))

        # 更新最佳验证集准确率和模型参数
        if epoch > 150:
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), args.save_path)  # 保存最佳模型参数
                early_stop_count = 0  # 重置早停计数器
            else:
                early_stop_count += 1

            # 判断是否触发早停策略
            if early_stop_count >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation accuracy."
                      f"Highest accuracy on validation is {best_val_acc:.4f}")
                break

    # 加载最佳模型参数并测试

    if args.MODEL_NAME in model_dict:
        model_class = model_dict[args.MODEL_NAME]
        best_model = model_class(input_dim, args.HIDDEN_DIM, output_dim, args.NUM_LAYERS, args.seed, args.dropout,
                                 args.USE_SAMPLING, args.Dynamic_Reorganization, args.WITH_BN).to(args.DEVICE)
    else:
        raise NotImplementedError("The model have not define!")

    best_model.load_state_dict(torch.load(args.save_path))
    best_model.eval().to(args.DEVICE)

    test_accuracy = ttt(best_model, data, args.PERCENT, ricci_curvature)
    file_path = args.accuracy_save_path
    # 写入测试准确率到 CSV 文件
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([test_accuracy])
    print("Test_Accuracy:{:.4f}".format(test_accuracy))
