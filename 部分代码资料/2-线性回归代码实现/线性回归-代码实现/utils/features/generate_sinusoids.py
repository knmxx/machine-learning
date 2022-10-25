import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    sin(x).
    """

    num_examples = dataset.shape[0]#样本数量
    sinusoids = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)#左右合并
        
    return sinusoids
