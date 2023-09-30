## 运行环境

- Ubuntu 20.04.1
- Python 3.10.13
- PyTorch 1.12.1

## 权重

权重在`model`文件夹中
* `model_1.ckpt`：第一次训练中第17,500次的权重，测试得分为283.30
* `model_2.ckpt`：第一次训练中第18,600次的权重，测试得分为-44.09
* `model_3.ckpt`：第二次训练中第13,300次的权重，测试得分为910.70
* `model_4.ckpt`：第二次训练中第16,800次的权重，测试得分为476.93
* `model_final.ckpt`：第二次训练中第20,000次的权重，为最终训练结果，测试得分为4833.60

## 训练参数
该项目中的权重基于以下参数训练获得

```python
UPDATE_MIN = 1              # 每次更新最少获取N条数据
RETAIN_COUNT = 2_000        # 数据池中最多保留的数据数量
LOG = False                 # 是否输出每一步的日志信息
TRAINING_COUNT = 10         # 每次更新数据后训练的次数
BATCH_SIZE = 512            # 每次训练的批量大小
EPOCH = 1000                # 总训练轮数
LOAD_MODEL = None           # 读取保存的模型参数
```

## 参考

- [markub3327/flappy-bird-gymnasium: An OpenAI Gym environment for the Flappy Bird game (github.com)](https://github.com/markub3327/flappy-bird-gymnasium/tree/main)
- [lansinuote/More_Simple_Reinforcement_Learning (github.com)](https://github.com/lansinuote/More_Simple_Reinforcement_Learning)