const tf = require('@tensorflow/tfjs');

// 创建一个序列化的模型
const model = tf.sequential();

// 添加第一层卷积层，输入尺寸为 [28, 28, 1] （图像大小和通道数）
// 有32个卷积核，大小为3x3，激活函数为relu
model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));

// 添加第二个卷积层，还是32个卷积核，大小为3x3，激活函数为relu
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));

// 添加最大池化层，池化窗口大小为2x2
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));

// 添加第三个卷积层，64个卷积核，大小为3x3，激活函数为relu
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));

// 添加第四个卷积层，还是64个卷积核，大小为3x3，激活函数为relu
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu',
}));

// 再次添加最大池化层
model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));

// 将所有的卷积层展平为一维向量
model.add(tf.layers.flatten());

// 添加Dropout层，防止模型过拟合
model.add(tf.layers.dropout({rate: 0.25}));

// 添加全连接层，512个神经元，激活函数为relu
model.add(tf.layers.dense({units: 512, activation: 'relu'}));

// 再次添加Dropout层
model.add(tf.layers.dropout({rate: 0.5}));

// 添加输出层（全连接层），10个神经元，激活函数为softmax
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

// 配置优化器和损失函数以及评估指标
const optimizer = 'rmsprop';
model.compile({
  optimizer: optimizer, // 使用 RMSProp 优化器
  loss: 'categoricalCrossentropy', // 使用分类交叉熵作为损失函数
  metrics: ['accuracy'], // 评估指标为准确率
});

module.exports = model;
