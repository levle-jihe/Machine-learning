import tensorflow as tf

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1, #  这里神经元是1，当神经元数量多的时候，每个神经元都会学习到一组系数和偏置。
            activation=None,
            # 权重初始化为0，
            kernel_initializer=tf.zeros_initializer(),
            # 偏置初始化为0
            bias_initializer=tf.zeros_initializer()
        )
    # 定义神经元
    def call(self, input):
        output = self.dense(input)
        return output

# 定义线性模型
model = Linear()
# 对权重做梯度下降
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)  # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    # 使用 model.variables 这一属性直接获得模型中的所有变量
    # model.variables是来源于父类吗？
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)

