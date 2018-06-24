from pspnet import PSPNetwork
from tensorflow.contrib.learn.python.learn.datasets import base

data_sets = base.Datasets(train=train, validation=None, test=test)
psp_net = PSPNetwork(
    data_sets=data_sets,
)
