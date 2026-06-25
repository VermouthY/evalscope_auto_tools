import argparse
import logging
from pathlib import Path

from evalscope.perf.main import run_perf_benchmark
from evalscope.perf.arguments import Arguments
from evalscope import TaskConfig, run_task

from config import *
from generate_dataset import *
from save_file import copy_log, save_log
from cal_prefix_hit_rate import *

logging.getLogger().setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=3500, help="input token length")
    parser.add_argument("--output_len", type=str, default="1500", help="output token length")
    parser.add_argument("--data_num", type=int, default=8192, help="dataset number")
    parser.add_argument("--concurrency", type=str, default="2048", help="max concurrency")
    parser.add_argument("--request_rate", type=str, default="-1", help="request rate")
    parser.add_argument("--dataset", type=str, default="none", help="dataset path")
    parser.add_argument("--repeat", type=int, default=1, help="number of test repeat times")
    parser.add_argument("--test_accuracy", action='store_true', default=False, help="test accuracy")
    parser.add_argument("--dataset_name", type=str, default="none", help="dataset name")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="The batch size for evaluation.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of tokens generated.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature, range 0~2")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling; only considers tokens accounting for top_p probability mass")
    parser.add_argument("--npu_num", type=int, default=1, help="npu numbers")
    parser.add_argument("--dataset_type", type=str, default="normal", help="normal or prefix_cache")
    parser.add_argument("--prefix_num", type=int, default=1, help="prefix numbers")
    parser.add_argument("--repeat_rate", type=str, default="0", help="dataset repeat rate")
    parser.add_argument("--prefix_test", action='store_true', default=False, help="test prefix dataset firstly")
    parser.add_argument("--seed", type=int, default=1, help="dataset random seed")
    parser.add_argument("--dp", type=int, default=1, help="dp size")
    parser.add_argument("--length_mean", type=int, default=None, help="gaussian mean for variable length")
    parser.add_argument("--length_std", type=float, default=None, help="gaussian std for variable length")
    parser.add_argument("--length_min", type=int, default=None, help="min length for uniform range or gaussian clip")
    parser.add_argument("--length_max", type=int, default=None, help="max length for uniform range or gaussian clip")
    return parser.parse_args()


def create_gsm8k_dataset(dataset_type, input_len, data_num, model_path, dataset_path, dp, prefix_num, repeat_rate, seed,
                         length_mean=None, length_std=None, length_min=None, length_max=None):
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"dataset work path {dataset_path} not exist")

    base_name = Path(model_path).name
    if dataset_type == "prefix_cache":
        prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,1,dp,repeat_rate,seed,prefix_num,
                                                                             length_mean, length_std, length_min, length_max)
        logging.info("[完成] 数据集已生成：")
        logging.info(f"  - 公共前缀：{prefix_jsonl_path}  (行数={dp*prefix_num})")
        logging.info(f"  - 数据集：  {dataset_jsonl_path} (行数={data_num})")
        logging.info("[信息] 配置：")
        logging.info(f"  tokens(单条长度)={input_len}, prefix_ratio(前缀重复率)={repeat_rate}")
        if length_mean is not None and length_std is not None:
            logging.info(f"  length_dist=gaussian(mean={length_mean}, std={length_std}, min={length_min}, max={length_max})")
        elif length_min is not None and length_max is not None:
            logging.info(f"  length_dist=uniform_int([{length_min}, {length_max}])")
        else:
            logging.info("  length_dist=fixed")
    else:
        dataset_name = "GSM8K-in" + str(input_len) + "-num" + str(data_num) + "-" + base_name + ".jsonl"
        logging.info(f"dataset_name: {dataset_name}")
        dataset_jsonl_path = Path(dataset_path) / dataset_name
        prefix_jsonl_path = ""
        # 判断数据集是否存在
        if not Path(dataset_jsonl_path).exists():
            logging.warning(f"Dataset {dataset_name} is not exist. Start create dataset")
            # create_data(input_len, data_num, model_path, dataset_path)
            prefix_jsonl_path, dataset_jsonl_path = create_multi_prefix_dataset(model_path,input_len,data_num,dataset_path,0,dp,0,seed,prefix_num,
                                                                                 length_mean, length_std, length_min, length_max)
            logging.info(f"Dataset {dataset_name} created.")
        else:
            logging.info(f"Dataset {dataset_name} exist.")
    return prefix_jsonl_path, dataset_jsonl_path


def generate_perf_command(dataset_path, input_len, output_len, concurrency, data_num, request_rate):
    task_cfg = Arguments(
        url=f'http://{HOST_IP}:{HOST_PORT}/v1/chat/completions',
        api='openai',
        model=MODEL_NAME,
        parallel=[concurrency],
        number=[data_num],
        rate=[request_rate],
        dataset='custom',
        dataset_path=str(dataset_path),
        min_prompt_length=input_len,
        max_prompt_length=input_len,
        min_tokens=output_len, # The minimum number of tokens that can be generated.
        max_tokens=output_len, # The maximum number of tokens that can be generated.
        tokenizer_path=MODEL_PATH,
        outputs_dir=OUTPUT_DIR,
        extra_args={'ignore_eos': True}
    )
    return task_cfg


def generate_eval_command(dataset_name, eval_batch_size, max_tokens, temperature, top_p):
    dataset_id = EVAL_DATASETS[dataset_name]
    task_cfg = TaskConfig(
        model=MODEL_NAME,
        api_url=f'http://{HOST_IP}:{HOST_PORT}/v1',
        eval_batch_size=eval_batch_size,
        datasets=[dataset_name],
        dataset_args={dataset_name:{'dataset_id':dataset_id}},
        generation_config={'max_tokens':max_tokens, 'temperature':temperature, 'top_p':top_p},
        work_dir=OUTPUT_DIR,
    )
    return task_cfg


def get_pod_metrics_info(pod_info):
    query_tokens, query_tokens_external,hit_tokens,hit_tokens_external = {},{},{},{}
    for pod in pod_info:
        ip,port = pod.split(":")
        query_tokens[pod],query_tokens_external[pod] = get_prefix_queries_total(ip,port)
        hit_tokens[pod], hit_tokens_external[pod] = get_prefix_hits_total(ip,port)
    return query_tokens, query_tokens_external, hit_tokens, hit_tokens_external


def cal_prefix_hit_info(query_tokens, query_tokens_external, hit_tokens, hit_tokens_external,
                        query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new):
    if not query_tokens or not query_tokens_external or not hit_tokens or not hit_tokens_external:
        return
    
    # 定义列宽
    col1_width = 15   # engine id
    col2_width = 20   # hbm hit rate
    col3_width = 20   # hbm(hit/query)
    col4_width = 20   # external hit rate
    col5_width = 20   # external(hit/query)
    
    # 按POD分组遍历
    for pod, engines in sorted(query_tokens.items()):
        # 准备数据行
        data_rows = []
        for engine_id, token in sorted(engines.items()):
            query_hbm = query_tokens_new[pod][engine_id] - query_tokens[pod][engine_id]
            hits_hbm = hit_tokens_new[pod][engine_id] - hit_tokens[pod][engine_id]
            query_ex = query_tokens_external_new[pod][engine_id] - query_tokens_external[pod][engine_id]
            hits_ex = hit_tokens_external_new[pod][engine_id] - hit_tokens_external[pod][engine_id]
            
            if query_hbm == 0:
                hit_rate_str = "0%"
                hit_detail = "0/0"
            else:
                hit_rate_str = format(hits_hbm / query_hbm, '.2%')
                hit_detail = f"{hits_hbm}/{query_hbm}"
            
            if query_ex == 0:
                hit_rate_ex_str = "0%"
                hit_ex_detail = "0/0"
            else:
                hit_rate_ex_str = format(hits_ex / query_ex, '.2%')
                hit_ex_detail = f"{hits_ex}/{query_ex}"
            
            data_rows.append({
                'engine_id': str(engine_id),
                'hbm_rate': hit_rate_str,
                'hbm_detail': hit_detail,
                'external_rate': hit_rate_ex_str,
                'external_detail': hit_ex_detail
            })
        
        # 定义表头
        headers = ['engine_id', 'hbm_hit_rate', 'hbm(hit/query)', 'external_hit_rate', 'external(hit/query)']
        
        # 计算总宽度
        total_width = col1_width + col2_width + col3_width + col4_width + col5_width + 8
        
        # 打印POD信息
        print("\n" + "=" * total_width)
        print(f"POD: {pod}")
        print("=" * total_width)
        
        # 打印表头
        print(f"{headers[0]:<{col1_width}} {headers[1]:<{col2_width}} {headers[2]:<{col3_width}} {headers[3]:<{col4_width}} {headers[4]:<{col5_width}}")
        print("-" * total_width)
        
        # 打印数据行
        for row in data_rows:
            print(f"{row['engine_id']:<{col1_width}} {row['hbm_rate']:<{col2_width}} {row['hbm_detail']:<{col3_width}} {row['external_rate']:<{col4_width}} {row['external_detail']:<{col5_width}}")
        
        print("=" * total_width)


def validate_length_args(args):
        # 变长参数校验
    if (args.length_mean is None) ^ (args.length_std is None):
        raise ValueError("length_mean 和 length_std 必须同时提供或同时不提供")
    if (args.length_min is None) ^ (args.length_max is None):
        raise ValueError("length_min 和 length_max 必须同时提供或同时不提供")
    if args.length_mean is not None and args.length_mean < 1:
        raise ValueError("length_mean 必须 >= 1")
    if args.length_std is not None and args.length_std < 0:
        raise ValueError("length_std 必须 >= 0")
    if args.length_min is not None and args.length_min < 1:
        raise ValueError("length_min 必须 >= 1")
    if args.length_max is not None and args.length_max < 1:
        raise ValueError("length_max 必须 >= 1")
    

def log_args(args):
    logging.info("============= Test Config =============")

    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")


def run_accuracy_test(args):
    task_cfg = generate_eval_command(
        args.dataset_name,
        args.eval_batch_size,
        args.max_tokens,
        args.temperature,
        args.top_p,
    )

    logging.info(f"eval test start, use command: {task_cfg}")
    run_task(task_cfg=task_cfg)
    result_log = Path(task_cfg.work_dir) / "logs" / "eval_log.log"
    dst_log = Path.cwd() / "evaluation.log"

    copy_log(result_log, dst_log)
    save_log(dst_log, "evaluation_all.log")


def run_perf_test(args):
    """
    性能测试主流程
    """

    repeat_rate = parse_prefix_ratio(args.repeat_rate)

    # 数据集准备
    if args.dataset == "none":
        src_file_prefix, src_file_data = create_gsm8k_dataset(
            args.dataset_type,
            args.input_len,
            args.data_num,
            MODEL_PATH,
            DATASET_PATH,
            args.dp,
            args.prefix_num,
            repeat_rate,
            args.seed,
            args.length_mean,
            args.length_std,
            args.length_min,
            args.length_max,
        )
    else:
        dataset_path = Path(args.dataset)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {args.dataset} does not exist.")

        src_file_data = str(dataset_path)
        src_file_prefix = ""

    # Prefix Cache测试
    if args.dataset_type == "prefix_cache":
        run_prefix_cache_test(args, src_file_prefix, src_file_data)

    # 普通性能测试
    else:
        run_normal_perf_test(args, src_file_data)


def run_normal_perf_test(args, dataset_path):
    logging.info("[开始] 全量数据集测试")

    task_cfg = generate_perf_command(
        dataset_path,
        args.input_len,
        args.output_len,
        args.concurrency,
        args.data_num,
        args.request_rate,
    )

    if args.repeat > 1:
        for test_time in range(args.repeat):
            logging.info(f"Execution rounds: {test_time + 1}")
            run_perf_benchmark(task_cfg)

    else:
        run_perf_benchmark(task_cfg)
    
    logging.info(f"性能测试结果原始路径: {task_cfg.outputs_dir}")
    save_benchmark_result(task_cfg)


def run_prefix_cache_test(
    args,
    src_file_prefix,
    src_file_data,
):
    if not POD_INFO:
        pod_info = [f"{HOST_IP}:{HOST_PORT}"]
    else:
        pod_info = POD_INFO

    logging.info(f"pod_info: {pod_info}")

    if args.prefix_test:
        warmup_prefix_cache(args, pod_info, src_file_prefix)

    logging.info("[开始] 全量数据集测试")

    query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

    task_cfg = generate_perf_command(
        src_file_data,
        args.input_len,
        args.output_len,
        args.concurrency,
        args.data_num,
        args.request_rate,
    )

    run_perf_benchmark(task_cfg)

    query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)

    cal_prefix_hit_info(
        query_tokens,
        query_tokens_external,
        hit_tokens,
        hit_tokens_external,
        query_tokens_new,
        query_tokens_external_new,
        hit_tokens_new,
        hit_tokens_external_new,
    )

    save_benchmark_result(task_cfg)


def warmup_prefix_cache(
    args,
    pod_info,
    prefix_dataset_path,
):
    logging.info("[开始] 前缀数据集预热")

    query_tokens, query_tokens_external, hit_tokens, hit_tokens_external = get_pod_metrics_info(pod_info)

    task_cfg = generate_perf_command(
        prefix_dataset_path,
        args.input_len,
        1,
        args.dp,
        args.dp * args.prefix_num,
        args.request_rate,
    )

    run_perf_benchmark(task_cfg)

    query_tokens_new, query_tokens_external_new, hit_tokens_new, hit_tokens_external_new = get_pod_metrics_info(pod_info)

    cal_prefix_hit_info(
        query_tokens,
        query_tokens_external,
        hit_tokens,
        hit_tokens_external,
        query_tokens_new,
        query_tokens_external_new,
        hit_tokens_new,
        hit_tokens_external_new,
    )


def save_benchmark_result(task_cfg):
    result_dir = task_cfg.outputs_dir

    src_file = Path(result_dir) / "benchmark.log"
    dst_file = Path.cwd() / "benchmark.log"

    copy_log(src_file, dst_file)
    save_log(dst_file, "benchmark_all.log")


def main():
    args = parse_arguments()

    validate_length_args(args)

    log_args(args)

    if args.test_accuracy:
        run_accuracy_test(args)
    else:
        run_perf_test(args)

 
if __name__ == '__main__':
    main()
    