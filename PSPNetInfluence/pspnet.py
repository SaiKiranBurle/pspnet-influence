import tensorflow as tf

from influence.genericNeuralNet import GenericNeuralNet
from model import PSPNet101, PSPNet50

MODEL_PATH = './ade20k_model/pspnet50/model.ckpt-0'


class PSPNetwork(GenericNeuralNet):

    def __init__(self, input_dim, output_dim, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(PSPNetwork, self).__init__(**kwargs)

        model_path = kwargs.get('model_path', MODEL_PATH)
        self.model = net = PSPNet50({'data': img}, is_training=False, num_classes=num_classes)

        self.load_model_from_path(model_path)

    def get_all_params(self):
        raise NotImplementedError

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, self.output_dim),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def inference(self):
        raise NotImplementedError

    def predictions(self):
        raise NotImplementedError

    def set_params(self):
        raise NotImplementedError

    # TODO: Should override loss?
