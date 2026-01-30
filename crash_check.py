#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志文件关键字扫描脚本
功能：扫描指定文件夹中的.log文件，查找包含"segmentation"或"internal"关键字的文件
修复了字节模式的正则表达式问题
"""

import os
import sys
import re
import mmap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
import argparse


class LogScanner:
    def __init__(self, keywords: List[str], case_sensitive: bool = False):
        """
        初始化日志扫描器
        
        Args:
            keywords: 要搜索的关键字列表
            case_sensitive: 是否区分大小写
        """
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        
        # 预编译正则表达式以提高性能
        # 使用字节模式(b'...')，因为mmap返回的是字节对象
        flags = 0 if case_sensitive else re.IGNORECASE
        
        # 为字节模式构建正则表达式
        pattern = b'|'.join(re.escape(keyword.encode('utf-8')) for keyword in keywords)
        self.regex = re.compile(pattern, flags)
        
        # 同时也保留字符串模式，用于普通文件读取
        str_pattern = '|'.join(re.escape(keyword) for keyword in keywords)
        self.str_regex = re.compile(str_pattern, flags)
    
    def scan_file(self, file_path: Path) -> bool:
        """
        扫描单个文件是否包含关键字
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 如果包含关键字返回True，否则返回False
        """
        try:
            # 尝试使用mmap以提高大文件读取性能
            with open(file_path, 'rb') as f:  # 以二进制模式打开
                try:
                    # 使用mmap内存映射
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                        # 在内存映射对象中搜索（字节模式）
                        return bool(self.regex.search(mmap_obj))
                except (ValueError, OSError):
                    # 如果mmap失败（如文件为空），回退到普通读取
                    f.seek(0)
                    content = f.read()
                    return bool(self.regex.search(content))
        except (IOError, OSError, ValueError) as e:
            # 对于无法处理的文件，尝试用文本模式读取
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    return bool(self.str_regex.search(content))
            except Exception:
                # 记录错误并跳过
                print(f"警告: 无法读取文件 {file_path}: {e}", file=sys.stderr)
                return False
    
    def scan_file_optimized(self, file_path: Path) -> bool:
        """
        优化的文件扫描方法，分块读取以节省内存
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 如果包含关键字返回True，否则返回False
        """
        try:
            with open(file_path, 'rb') as f:  # 以二进制模式打开
                # 读取前几MB，如果文件很大
                chunk_size = 1024 * 1024  # 1MB
                
                # 先读取第一个块
                chunk = f.read(chunk_size)
                if not chunk:
                    return False
                
                # 检查第一个块
                if self.regex.search(chunk):
                    return True
                
                # 如果文件很大，继续读取剩余部分
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    if self.regex.search(chunk):
                        return True
                
                return False
                
        except (IOError, OSError) as e:
            # 回退到文本模式
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # 分块读取文本
                    chunk_size = 1024 * 1024
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        if self.str_regex.search(chunk):
                            return True
                    return False
            except Exception:
                print(f"警告: 无法读取文件 {file_path}: {e}", file=sys.stderr)
                return False
    
    def scan_directory(self, directory: Path, max_workers: int = 8) -> List[Path]:
        """
        扫描目录中的所有.log文件
        
        Args:
            directory: 目录路径
            max_workers: 最大线程数，固定为8
            
        Returns:
            List[Path]: 包含关键字的文件路径列表
        """
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"无效的目录路径: {directory}")
        
        # 收集所有.log文件
        log_files = list(directory.rglob("*.log"))
        print(f"找到 {len(log_files)} 个.log文件")
        
        if not log_files:
            return []
        
        matching_files = []
        processed = 0
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.scan_file_optimized, file_path): file_path 
                for file_path in log_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                processed += 1
                file_path = future_to_file[future]
                
                try:
                    if future.result():
                        matching_files.append(file_path)
                except Exception as e:
                    print(f"错误: 处理文件 {file_path} 时发生异常: {e}", file=sys.stderr)
                
                # 显示进度
                if processed % 100 == 0 or processed == len(log_files):
                    print(f"进度: {processed}/{len(log_files)} 文件已处理", end='\r')
        
        print()  # 换行
        return matching_files
    
    def save_results(self, matching_files: List[Path], output_file: Path):
        """
        保存结果到文件
        
        Args:
            matching_files: 匹配的文件路径列表
            output_file: 输出文件路径
        """
        # 按文件名排序
        matching_files.sort(key=lambda x: x.name)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 扫描时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 扫描目录: {matching_files[0].parent if matching_files else 'N/A'}\n")
            f.write(f"# 关键字: {', '.join(self.keywords)}\n")
            f.write(f"# 线程数: 8 (固定)\n")
            f.write(f"# 匹配文件数: {len(matching_files)}\n")
            f.write("=" * 50 + "\n\n")
            
            for file_path in matching_files:
                # 写入相对路径或绝对路径
                f.write(f"{file_path.absolute()}\n")
        
        print(f"结果已保存到: {output_file.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="扫描.log文件中的关键字",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /var/log                            # 扫描/var/log目录
  %(prog)s /path/to/logs -o results.txt        # 指定输出文件
  %(prog)s . -t 4                              # 使用4个线程扫描当前目录
  %(prog)s /var/log -c                         # 区分大小写扫描
        """
    )
    
    parser.add_argument(
        "directory",
        help="要扫描的目录路径"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="matching_logs.txt",
        help="输出文件路径 (默认: matching_logs.txt)"
    )
    
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=8,
        help="线程数 (默认: 8，固定为8线程)"
    )
    
    parser.add_argument(
        "-k", "--keywords",
        nargs='+',
        default=["segmentation", "internal"],
        help="要搜索的关键字 (默认: segmentation internal)"
    )
    
    parser.add_argument(
        "-c", "--case-sensitive",
        action="store_true",
        help="区分大小写 (默认: 不区分)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    # 验证目录
    directory = Path(args.directory)
    if not directory.exists():
        print(f"错误: 目录 '{directory}' 不存在")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"错误: '{directory}' 不是目录")
        sys.exit(1)
    
    # 创建扫描器
    scanner = LogScanner(
        keywords=args.keywords,
        case_sensitive=args.case_sensitive
    )
    
    print("=" * 60)
    print("日志文件扫描器")
    print("=" * 60)
    print(f"扫描目录: {directory.absolute()}")
    print(f"关键字: {', '.join(args.keywords)}")
    print(f"大小写敏感: {'是' if args.case_sensitive else '否'}")
    print(f"输出文件: {args.output}")
    print(f"线程数: {args.threads} (固定)")
    print("-" * 60)
    
    try:
        # 开始扫描
        start_time = time.time()
        matching_files = scanner.scan_directory(directory, args.threads)
        elapsed_time = time.time() - start_time
        
        # 保存结果
        if matching_files:
            scanner.save_results(matching_files, Path(args.output))
            
            # 显示匹配的文件
            if args.verbose:
                print("\n匹配的文件:")
                for i, file_path in enumerate(matching_files, 1):
                    print(f"  {i:3d}. {file_path}")
        else:
            print("未找到包含关键字的文件")
        
        # 显示统计信息
        print(f"\n扫描完成!")
        print(f"总文件数: {len(list(directory.rglob('*.log')))}")
        print(f"匹配文件数: {len(matching_files)}")
        print(f"处理时间: {elapsed_time:.2f} 秒")
        
    except KeyboardInterrupt:
        print("\n\n扫描被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()