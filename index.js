const tf = require('@tensorflow/tfjs-node');
const argparse = require('argparse');

const data = require('./data/index'); // 导入数据集加载器
const model = require('./model/index'); // 导入手写数字识别模型

async function run(epochs, batchSize, modelSavePath) {
  await data.loadData(); // 异步加载训练和测试数据集

  const {images: trainImages, labels: trainLabels} = data.getTrainData();
  
  // 打印模型摘要信息
  model.summary();

  let epochBeginTime;
  let millisPerStep;

  const validationSplit = 0.15; // 用于验证的划分比例
  const numTrainExamplesPerEpoch =
      trainImages.shape[0] * (1 - validationSplit); // 每个epoch的训练样例数
  const numTrainBatchesPerEpoch =
      Math.ceil(numTrainExamplesPerEpoch / batchSize); // 每个epoch的训练批次数

  // 使用异步fit()方法来训练模型
  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit
  });

  // 加载测试数据集，并对模型进行评估
  const {images: testImages, labels: testLabels} = data.getTestData();
  const evalOutput = model.evaluate(testImages, testLabels);

  console.log(
      `\nEvaluation result:\n` +
      `  Loss = ${evalOutput[0].dataSync()[0].toFixed(3)}; `+
      `Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`);

  // 如果指定了模型保存路径，则将训练好的模型保存到指定路径
  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`);
    console.log(`Saved model to path: ${modelSavePath}`);
  }
}

const parser = new argparse.ArgumentParser({
  description: 'TensorFlow.js-Node MNIST Example.',
  add_help: true
});

// 添加命令行参数
parser.add_argument('--epochs', {
  type: 'int',
  default: 20,
  help: 'Number of epochs to train the model for.'
});
parser.add_argument('--batch_size', {
  type: 'int',
  default: 128,
  help: 'Batch size to be used during model training.'
})
parser.add_argument('--model_save_path', {
  type: 'string',
  help: 'Path to which the model will be saved after training.'
});
const args = parser.parse_args();

// 运行
run(args.epochs, args.batch_size, args.model_save_path);
