import torch


def random_sparse_layer(edge_index, percent):
    """
    随机采样一定比例的边
    :param edge_index:
    :param percent:
    :return: 采样后的边的index
    """
    assert 0 <= percent <= 1, "Percent should be between 0 and 1."

    num_edges = edge_index.size(1)  # 获取边的总数

    num_preserve = int(num_edges * percent)  # 需要保留的边的数量

    # 随机抽样保留边的索引
    sampled_indices = torch.randperm(num_edges)[:num_preserve]

    # 根据抽样的索引获取保留的边
    preserved_edge_index = edge_index[:, sampled_indices]


    return preserved_edge_index

