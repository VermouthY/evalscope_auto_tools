# evalscope_auto_tools
 An automated testing tool based on EvalScope that facilitates performance and accuracy testing.

PR合入联系王晨阳

## 最新更新（2026/6/25）
1、支持变长数据集

2、bug修复repeat_rate 1场景

## 适用场景

- 模型性能测试
  - 一般性能测试
  - prefix cache性能测试
  - 指定gsm8k格式数据集的性能测试，如medium2

- 模型精度测试
  - 工具自带数据集精度测试
  - 其他数据集精度测试

- 利用本工具生成数据集

## 使用方法

进入带[evalscope](https://github.com/modelscope/evalscope)的环境，下载本工具 -> 修改本工具的config.py -> 运行python命令

### 一、修改config.py

**注：脚本会自动生成数据集，需先创建存放生成数据集的文件夹，eg：mkdir /mnt/path_to_store_dataset**

```python
# 数据集文件夹路径，需可访问
DATASET_PATH = "/home/dataset"

# 服务化配置的模型名称
MODEL_NAME = "dsv4"
# 模型权重路径, 用于读取 tokenizer
MODEL_PATH = "/home/weights/model_weights"
# 请求目的 IP
HOST_IP = "141.xx.xx.xx"
# 请求目的端口
HOST_PORT = "xxxx"

# evalscope输出日志保存路径
OUTPUT_DIR = "./outputs/default"

# 各节点信息，格式为 ["{ip}:{port}"]
# 用于查询vllm metrics计算各个dp域的prefix cache命中率，不配置默认为HOST_IP:HOST_PORT
# PD分离场景请填写各个节点的IP和对应dp域的port
# POD_INFO = ["141.xx.xx.11:8000","141.xx.xx.12:8000"]
POD_INFO = []

# 本地数据集路径，若精度测试的数据集不在下列数据集内，请先下载数据集并按照格式自行添加到以下字典中；
# 性能测试无需改动
EVAL_DATASETS={'mmlu_pro':'./eval_datasets/MMLU-Pro',
               'aime25':'./eval_datasets/aime25',
               'gpqa_diamond':'./eval_datasets/gpqa_diamond',
               'gsm8k':'./eval_datasets/gsm8k'}

```

### 二、参数详解

`python3 evalscope_test.py --help`可查看所有参数

| 参数名| type |释义 |
| --- | --- | --- |
| --input_len | int|性能测试输入长度|
| --output_len | int|性能测试输出长度 |
| --data_num | int | 数据集条数 |
| --concurrency | int |系统最大并发数 |
| --request_rate | int |请求频率，默认-1  |
| --dataset | str | 数据集路径，可指定测试数据集，仅限gsm8k格式                  |
| --repeat | int   | 单条命令的测试次数，默认值1，注：数据采集功能只能采集最后一次测试数据 |
| --test_accuracy | bool  | 开启精度测试，默认值false                                    |
| --dataset_name | str | 精度测试的数据集名称，例如mmlu_pro、aime25...（精度测试时使用） |
| --eval_batch_size | int | 同时进行精度评测的批量大小（精度测试时使用）                 |
| --max_tokens | int | 最大生成token数量（精度测试时使用）                          |
| --temperature | float | 采样温度，范围0~2，越高越随机（精度测试时使用）              |
| --top_p | float | Nucleus采样，考虑概率质量为top_p的token（精度测试时使用）    |
| --npu_num | int | npu卡数，用于计算单卡吞吐，默认值1 |
| --dataset_type| str | normal or prefix_cache，一般数据集or带前缀数据集，默认值normal|
| --prefix_num | int | 前缀个数，默认值1 |
| --repeat_rate| str | 数据集前缀重复率，默认值0.5，支持格式：百分比如 "50%" 或小数如 "0.5" |
| --prefix_test| bool | 是否在全量数据集测试前，先测试一遍前缀，默认值false|
| --seed| int | 随机种子，仅适用生成带前缀数据集，不同seed生成的随机token不重复，默认值1 |
| --dp| int | dp域数量，默认值1 ，保证模型推理时前缀会预热到每个dp域上|
| --length_mean | int|输入长度均值|
| --length_std | int|输入长度标准差|
| --length_min | int|输入长度最小值|
| --length_max | int|输入长度最大值|

### ※ 数据集生成逻辑

前缀：随机挑选一条未使用的GSM8K数据，重复/截取到输入长度input_len

后缀：随机挑选一条GSM8K数据（可能重复），重复/截取到固定长度

完整数据集 = 前缀 + 3个随机token（--seed控制） + 后缀

### ※ 前缀数据集结构说明

1、前缀重复率50%：指每条请求的前50%能在kv cache中命中

```
# 数据集示例
abc123
abc456
abc789
...
```

2、前缀个数：指前缀种类

```
# 前缀重复率50%，2个前缀数据集示例
abc123
abc456
def789
def!@#
...
```

3、随机种子seed：seed不同则生成的随机token不同

```
# 前缀重复率50%，前缀个数1，不同seed的数据集示例

# seed=1
${prefix_data1}qwe${suffix_data1}
${prefix_data1}rty${suffix_data2}
...

# seed=2
${prefix_data2}!@#${suffix_data1}
${prefix_data2}%^&${suffix_data2}
...
```

## 三、命令示例

#### 3.1 性能测试

**注：测试prefix cache功能需先预热前缀，保证跑完整数据集时存在命中**

1、测试2k/2k不带前缀的gsm8k数据集性能

```bash
python3 evalscope_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10
```

2、测试2k/2k带前缀的gsm8k数据集性能，前缀个数1，数据集前缀重复率50%，dp 2，**先预热前缀**

```bash
python3 evalscope_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 --dataset_type prefix_cache --repeat_rate 0.5 --prefix_test --dp 2
```

3、测试2k/2k带前缀的gsm8k数据集性能，前缀个数3，数据集前缀重复率73%，不预热前缀直接跑完整数据集

```bash
python3 evalscope_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 --dataset_type prefix_cache --repeat_rate 73% --seed 200 --prefix_num 3
```

4、测试8k~128k**不定长**，平均32k，带前缀的gsm8k数据集性能，数据集前缀重复率90%，dp 2，**先预热前缀**

```bash
python3 evalscope_test.py --input_len 32768 --output_len 300 --data_num 32 --concurrency 8 --request_rate -1 --dataset_type prefix_cache --repeat_rate 90% --prefix_test --dp 2 --length_mean 32768 --length_std 49152 --length_min 8192 --length_max 131072
```

5、测试指定数据集性能（仅限**gsm8k**格式）

```bash
python3 evalscope_test.py --dataset "/mnt/path_to_dataset/medium2.jsonl" --output_len 20 --concurrency 1024
```

#### 3.2 精度测试

- 目前该工具内自带`mmlu_pro`、`aime25`、`gpqa_diamond`、`gsm8k`四个常用数据集，若需测试其他数据集，请先下载到本地后，再把该数据集名称和本地路径作为键值对加入到`config.py`中的`EVAL_DATASETS`字典中。

- 该工具支持的数据集和对应下载链接请参考[evalscope](https://evalscope.readthedocs.io/zh-cn/latest/get_started/supported_dataset/index.html)工具官方文档。

举例：测试aime2025数据集

```bash
python3 evalscope_test.py --test_accuracy --dataset_name aime25 --eval_batch_size 8 --max_tokens 65536 --temperature 1.0 --top_p 1.0
```

## 四、结果获取

#### 4.1 性能测试日志

##### 性能测试结果日志：

`benchmark.log`

##### 已运行的所有性能测试的evalscope日志：

`benchmark_all.log` 

##### prefix cache命中率获取：

见打屏日志

#### 4.2 精度测试日志

##### 精度测试结果日志：

`evaluation.log`

##### 已运行的所有精度测试的evalscope日志：

`evaluation_all.log` 

## FAQ（常见问题）

### 1、出现ERROR日志：生成数据集失败，请清空picked ids

解决方案：删除picked_ids.txt文件

### 2、加载tokenizer报错

解决方案：检查当前transformers版本是否适配模型，如GLM5需更新mindie/vllm镜像内transformers版本

### 3、常用shell固定并发测试脚本

```bash
bs=(1 8 16 24 32 40 48 56)
for i in ${bs[@]}
do
        python3 evalscope_test.py --input_len 8192 --output_len 1 --data_num $(($i * 4)) --concurrency $i --request_rate 0 --dataset_type prefix_cache --repeat_rate 0.5  --seed $i --prefix_num 1 --prefix_test
done
```

### 4、打屏不显示prefix cache命中率信息

只有开启--prefix_test才会打印命中率信息，若开启后打屏不显示，可按照下面方案尝试解决。

------------解决方案分界线------------

1）检查config.py里的POD_INFO是否配置正确。

PD混部：配置主节点的IP和PORT。

PD分离：单开prefix cache时，配置P节点的IP和对应DP域的PORT；开启池化时，配置每个节点的IP和对应DP域的PORT

2）按照以下步骤手动验证是否能正确获取vllm metrics信息

a. 关闭proxy代理
```bash
unset http_proxy
unset https_proxy
```
b. 在测试前后分别发送curl命令获取metrics信息
```bash
curl -s http://{ip_address}:{port}/metrics | grep prefix
```
正常情况返回vllm:prefix_cache_queries和vllm:prefix_cache_hits信息，自行计算命中率，计算公式如下：

hit rate = (第二次查询prefix_cache_hits - 第一次查询prefix_cache_hits) / (第二次查询prefix_cache_queries - 第一次查询prefix_cache_queries)
