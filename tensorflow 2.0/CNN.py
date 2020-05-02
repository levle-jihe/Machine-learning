# _*_ coding:utf-8 _*_
# author: jihe
# time: 2020/5/2 16:05
# name: CNN
# From
import tensorflow as tf
import numpy as np


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

    # 模型的继承


# **********************************************模型更改**********************************************
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积层神经元（卷积核）数目 卷积之后的结果求均值为最后结果？？？？？？
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)  # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


# ***************************************************************************

num_epochs = 5
batch_size = 50
learning_rate = 0.001
#  更改！！！！！！！！！！！！！！！！！！
model = CNN()  # 卷积神经网络
# ！！！！！！！！！！！！！！！！！！！！
data_loader = MNISTLoader()  # 数据加载
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 当我们求到了损失  计算了梯度 数据优化交给optimizer Adam是算法
# 公式含义 每次训练50张，所有图片训练5次，需要训练多少次
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)  # 交叉熵损失函数
        #
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # model.variables-grads变量减去梯度

# 模型的评估
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 建评估器
#  载入测试集num_test_data 50张图片一个batch进行测试
num_batches = int(data_loader.num_test_data // batch_size)  #
for batch_index in range(num_batches):
    # 偏移量*batch量， 取50张图片 切片
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    # 预测的结果
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    # update_state记录评估器的结果
    '''sparse_ 该前缀作用是可以把[1, 5, 4],转换成01的向量
    '''
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
# result() 方法输出最终的评估指标值（预测正确的样本数占总样本数的比例）
print("test accuracy: %f" % sparse_categorical_accuracy.result())
