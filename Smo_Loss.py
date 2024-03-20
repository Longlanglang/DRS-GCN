import time
import config
import torch
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"



def  get_class_adj(node_num, labels):
    """
    用以根据标签生成一个类别矩阵
    :param node_num: 节点数量
    :param labels: 标签
    :return: 返回inter_class_adj代表所有不同类节点的集合，对应位置为1代表横纵索引指向的节点之间属于不同类别
             与之对应的intra_class_adj代表同类别节点之间的连接
    """

    # intra class
    labels_repeat = labels.repeat(node_num, 1).to(device)
    intra_class_adj = torch.zeros([node_num, node_num], device=device)
    intra_class_adj[labels_repeat == labels_repeat.T] = 1
    ones_matrix = torch.ones_like(intra_class_adj, device=device)
    diagonal_one_matrix = torch.eye(intra_class_adj.shape[0], device=device)
    diagonal_zero_matrix = ones_matrix - diagonal_one_matrix
    intra_class_adj = intra_class_adj * diagonal_zero_matrix
    inter_class_adj = 1 - intra_class_adj
    inter_class_adj = (inter_class_adj * diagonal_zero_matrix)
    return intra_class_adj, inter_class_adj

def get_smo_loss(representation, intra_class_adj, inter_class_adj):
    """
    用以计算节点之间的平滑度
    :param representation: 输入节点表示作为计算节点相似度的依据
    :param intra_class_adj: 同类别节点矩阵
    :param inter_class_adj: 不同类比额节点矩阵
    :return: SMO_inter_class 表征所有不同类别节点之间相似度的均值,
             SMO_intra_class 表征所有相同类别节点之间相似度的均值
    """
    representation_normalized = representation / (representation.norm(2, 1).reshape(-1, 1) + 1e-10)

    SMO_Matrix = 1 - representation_normalized @ representation_normalized.T

    SMO_intra_class = (MAD_Matrix * intra_class_adj).sum() / ((intra_class_adj).sum()).to(device)
    SMO_inter_class = (MAD_Matrix * inter_class_adj).sum() / ((inter_class_adj).sum()).to(device)

    return SMO_inter_class, SMO_intra_class

def get_global_SMO(representation):
    representation_normalized = representation / (representation.norm(2, 1).reshape(-1, 1) + 1e-10)
    MAD_Matrix = 1 - representation_normalized @ representation_normalized.T
    # 找到非零元素的索引
    non_zero_indices = torch.nonzero(MAD_Matrix)

    # 提取非零元素的值
    non_zero_values = MAD_Matrix[non_zero_indices[:, 0], non_zero_indices[:, 1]]

    # 计算非零元素的和
    non_zero_sum = torch.sum(non_zero_values)

    # 计算非零元素的数量
    non_zero_count = len(non_zero_values)

    # 计算非零均值
    mean_value = non_zero_sum / non_zero_count
    return mean_value
