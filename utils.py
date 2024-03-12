from Smo_Loss import get_smo_loss, get_class_adj
import torch
import time
import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def train(model, data, optimizer, criterion, percent, ricci_curvature, use_SMO, alpha1, alpha2):
    """
    训练函数主体
    :param model:
    :param data:
    :param optimizer:
    :param criterion:
    :param percent:
    :param use_MAD:
    :return: 损失函数和准确率
    """
    model.train()
    optimizer.zero_grad()
    out, representation = model(data.x, data.edge_index, percent, ricci_curvature)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    if use_SMO:
        # compute smoothness loss
        representation = (sum(representation) / len(representation)).to(DEVICE)

        intra_class_adj, inter_class_adj = get_class_adj(data.y[data.train_mask].shape[0],
                                                         data.y[data.train_mask])


        SMO_inter_class, SMO_intra_class = get_smo_loss(representation[data.train_mask], intra_class_adj,
                                                        inter_class_adj)

        # alpha1 = 0.2
        # alpha2 = 0.02
        loss += alpha1 * SMO_intra_class - alpha2 * SMO_inter_class
    loss.backward()

    optimizer.step()

    pred = out.argmax(dim=1)
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_accuracy = int(train_correct.sum()) / int(data.train_mask.sum())
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    val_accuracy = int(val_correct.sum()) / int(data.val_mask.sum())
    return loss, train_accuracy, val_accuracy


"""     num = len(representation)
        MAD_inter_class, MAD_intra_class = 0, 0
        intra_class_adj, inter_class_adj = get_class_adj(data.y[data.train_mask].shape[0], data.y[data.train_mask])
        for i in range(num):
            MAD_inter_class_new, MAD_intra_class_new = get_mad_loss(representation[i][data.train_mask], intra_class_adj, inter_class_adj)
            MAD_inter_class += MAD_inter_class_new
            MAD_intra_class += MAD_intra_class_new
"""

def ttt(model, data, percent, ricci_curvature):
    """
    测试函数主体
    :param model:
    :param data:
    :param percent:
    :return: 返回识别准确率
    """
    model.eval()
    out, _ = model(data.x, data.edge_index, percent, ricci_curvature)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc