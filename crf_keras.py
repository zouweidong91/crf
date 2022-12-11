# -*- coding:utf-8 -*-

from keras.layers import Layer
import keras.backend as K


class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def log_norm_step(self, inputs, states):
        """递归计算归一化因子  rnn中cell定义,每一个实timestep，调用该函数计算,更新outputs和states
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        中华人民共和国 
        初始的states[0]就是句子中第一个单词：中 (batch_size,output_dim)  通过expand_dims 转换为(batch_size, output_dim, 1)
        rnn过程中不断输入后续句子inputs，更新state。例如：华 (batch_size, output_dim), 通过expand_dims 转换为(batch_size, 1, output_dim)
        trans 通过expand_dims 转换为(1, output_dim, output_dim)
        states[0] + inputs + trans 就得到一次计算的outputs，再做后续mask处理
        """
        inputs, mask = inputs[:, :-1], inputs[:, -1:]
        states = K.expand_dims(states[0], 2)  # (batch_size, output_dim, 1)   output_dim=num_label
        trans = K.expand_dims(self.trans, 0)  # (1, output_dim, output_dim)
        outputs = K.logsumexp(states + trans, 1)  # (batch_size, output_dim)
        outputs = outputs + inputs
        outputs = mask * outputs + (1 - mask) * states[:, :, 0]
        return outputs, [outputs]

    def path_score(self, inputs, labels):  # inputs:y_pred  labels:y_true
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。因为其中labels是one-hot编码，点乘本质就是取目标路径
        (batch_zise, seq_len, num_label)
        中华人民共和国
        exp:
            labels= np.arange(24).reshape(2,3,4)
            labels1 * labels2:
                <tf.Tensor: id=76, shape=(2, 2, 4, 4), dtype=int64, numpy=
                array([[[[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1]],  # 改矩阵显示的是prev_word-->obs_word转移过程中命中的trans矩阵的位置

                        [[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1]]],


                    [[[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1]],

                        [[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1]]]])>

        tf.einsum更简洁实现见bert4keras源码
        """
        point_score = K.sum(K.sum(inputs * labels, 2), 1, keepdims=True)  # 逐标签得分  点乘， 
        labels1 = K.expand_dims(labels[:, :-1], 3)  # 中华人民共和  (2, 2, 4, 1)
        labels2 = K.expand_dims(labels[:, 1:], 2)   #  华人民共和国 (2, 2, 1, 4)
        labels = labels1 * labels2  # 两个错位labels，负责从转移矩阵中抽取目标转移得分  (2, 2, 4, 4)
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)  # (1, 1, 4, 4)
        trans_score = K.sum(K.sum(trans * labels, [2, 3]), 1, keepdims=True)  
        return point_score + trans_score  # 两部分得分之和

    def call(self, inputs):  # CRF本身不改变输出，它只是一个loss
        return inputs

    def loss(self, y_true, y_pred):  # y_true是one hot形式
        if self.ignore_last_label:
            mask = 1 - y_true[:, :, -1:]
        else:
            mask = K.ones_like(y_pred[:, :, :1])
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        path_score = self.path_score(y_pred, y_true)  # 计算分子（对数）
        init_states = [y_pred[:, 0]]  # 初始状态, 第一个单词
        y_pred = K.concatenate([y_pred, mask])
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states)  # 计算Z向量（对数） 计算所有路径的总分。借用rnn迭代计算
        log_norm = K.logsumexp(log_norm, 1, keepdims=True)  # 计算Z（对数）
        return log_norm - path_score  # 即-log(分子/分母)

    def accuracy(self, y_true, y_pred):  # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1 - y_true[:, :, -1] if self.ignore_last_label else None
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal * mask) / K.sum(mask)
