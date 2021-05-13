# Timeseries Forecast

# Dataset
## Electricity consumption
原始数据：https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014。

用电量数据，单位kw/h，一共有321个用户从2011年到2014年每15分钟采样一次的数据。
处理：去掉2011年数据（大部分为0）以及重采样为以小时为粒度的序列数据。

数据规模：26304 x 321， 321个序列，每个序列长度在26304，即3年每小时的用电量，各个序列值差别较大

## Traffic Usage
原始数据：http://pems.dot.ca.gov

交通流量数据，来自加利佛利亚交通运输部分，收集了2015年到2016年一共48个月以小时为粒度的序列数据，一共x个传感器。
原始数据在0-1之间，表示道路占用率。

数据规模：17544 x 862, 862个序列，每个序列长度17544，即2年每小时交通数据，各个序列值进行了归一化，差别不大


## Solar Energy
原始数据：http://www.nrel.gov/grid/solar-power-data.html

太阳能数据，记录137个植物2005年每10分钟采样数据

数据规模：52560 x 137, 137个序列，每个序列长度52560，即1年每10分钟交通数据,

## Exchange Rate

8个国家的汇率变化情况，1990到2016年每天的变化情况。Australia, British, Canada, Switzerland, China, Japan, New Zealand and Singapore ranging

数据规模：7588 x 8, 8个序列，每个序列长度7588，即26年每天的数据, 比较符合小规模场景

## ETTh1
2年时间范围内电力数据，主要应用于long input 和 long output场景下

数据规模：17420 x 8,


# Result
## LSTNet
```
Traffic数据集：
    ==> Epochs: 70, Valid Loss: 22.80272, Valid RSE: 0.40866
    Epochs: 72, Iterations: 5913, Train Loss: 34.45109
    Epochs: 74, Iterations: 6075, Train Loss: 34.49729
    Start validate
    ==> Epochs: 75, Valid Loss: 22.31715, Valid RSE: 0.40429
    ==> Early Stop!
    Start test
    Test shape: (3509, 862), (3509, 862)
    Test metric: 0.40069
```

## Informer
```
ETTh1数据集：
    Epoch: 4 cost time: 14.527852058410645
    Epoch: 4, Steps: 266 | Train Loss: 0.1711380 Vali Loss: 0.6871772 Test Loss: 0.6032237
    EarlyStopping counter: 3 out of 3
    Early stopping
    test 2857
    test shape: (89, 32, 24, 7) (89, 32, 24, 7)
    test shape: (2848, 24, 7) (2848, 24, 7)
    mse:0.5284854769706, mae:0.5319210290908
```

