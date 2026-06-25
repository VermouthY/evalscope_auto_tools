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
