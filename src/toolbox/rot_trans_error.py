import torch
import numpy as np
@torch.no_grad()



def average_translation_distances(translations1, translations2,measurement='MSE'):
    # 计算每个对应平移向量之间的欧氏距离
    # distances = np.linalg.norm(translations1 - translations2, axis=-1)/3
    # distances = np.mean(np.abs(translations1 - translations2), axis=-1)
    # 计算每个批次的平均距离
    # average_distances = np.mean(distances, axis=1)
    if measurement == 'MSE':
        average_distances = np.mean((translations1-translations2)**2,axis=(-1,-2))
    else:
        average_distances = np.mean(np.abs(translations1-translations2),axis=(-1,-2))

    # print((translations1 - translations2)[1,:,0])
    # print('='*10)
    # print((translations1 - translations2)[1,:,1])
    # print('='*10)
    # print((translations1 - translations2)[1,:,2])
    # print(average_distances)
    # exit()
    # distances = (translations1 - translations2)
    # average_distances = np.mean(distances**2, axis=(-1,-2))
    return average_distances

def quaternion_similarity(q1, q2):
    """
    计算两个四元数的相似度。返回余弦相似度和旋转角度差异。
    q1, q2: 四元数，形状为 (..., 4)
    """
    # 确保四元数是单位四元数
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    q2 = q2 / q2.norm(dim=-1, keepdim=True)
    
    # 计算点积
    dot_product = torch.sum(q1 * q2, dim=-1)
    
    # 计算余弦相似度
    cosine_similarity = torch.abs(dot_product)
    
    # 计算旋转角度差异
    angle_difference = 2 * torch.acos(cosine_similarity)
    
    return cosine_similarity, angle_difference

@torch.no_grad()
def quaternion_distance(q1, q2):
    dot_product = np.einsum('...i,...i', q1, q2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angles = 2 * np.arccos(np.abs(dot_product))
    angles_degrees = np.degrees(angles)
    return angles_degrees

@torch.no_grad()
def normalize_quaternions(quaternions):
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return quaternions / norms

@torch.no_grad()
def average_quaternion_distances(batch_quaternions1, batch_quaternions2):
    batch_quaternions1 = normalize_quaternions(batch_quaternions1)
    batch_quaternions2 = normalize_quaternions(batch_quaternions2)
    num_batches, num_quaternions, _ = batch_quaternions1.shape
    average_distances = np.zeros(num_batches)

    for i in range(num_batches):
        distances = quaternion_distance(batch_quaternions1[i], batch_quaternions2[i])
        average_distances[i] = np.mean(distances)

    return average_distances
