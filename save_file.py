import logging
from datetime import datetime
from pathlib import Path
import shutil
logging.getLogger().setLevel(logging.INFO)


def copy_log(src_file, dst_file):
    if not src_file.exists():
        raise FileNotFoundError(src_file)
    
    # 如果目标文件已存在，先删除
    if dst_file.exists():
        dst_file.unlink()

    # 复制文件
    shutil.copy(src_file, dst_file)
    return dst_file


def save_log(source_file, target_file):
    try:
        # 读取源文件内容
        with open(source_file, 'r', encoding='utf-8') as src:
            content = src.read()
        
        # 追加到目标文件
        with open(target_file, 'a', encoding='utf-8') as tgt:
            # 添加分隔线和时间戳
            tgt.write(f"\n\n{'='*50}\n")
            tgt.write(f"{'='*50}\n\n")
            tgt.write(content)
            tgt.write(f"\n\n{'='*50}\n")
            tgt.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            tgt.write(f"{'='*50}\n")
        
        logging.info(f"成功将 {source_file} 的内容追加到 {target_file}")
        
    except FileNotFoundError:
        logging.error(f"错误：找不到文件，请检查文件路径是否正确")
    except Exception as e:
        logging.error(f"发生错误：{e}")
