#!/usr/bin/env python3
import argparse
import yaml
import time
import subprocess
import shutil
from pathlib import Path
from llm_model_ex import LLMGenerator 
from tqdm import tqdm

# Mapping from compiler to file extension and programming language
COMPILER_INFO = {
    "java": {"ext": ".java", "lang": "java"},
    "gcc": {"ext": ".c", "lang": "c"},
    "clang": {"ext": ".c", "lang": "c"},
    "g++": {"ext": ".cpp", "lang": "cpp"},
    "go": {"ext": ".go", "lang": "go"},
    "cvc5": {"ext": ".smt2", "lang": "smt2"}
}

def append_log(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def format_hms(total_seconds: int) -> str:
    """
    Format seconds as XhYminZs, YminZs, or Zs.
    """
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}h{minutes}min{seconds}s"
    if minutes > 0:
        return f"{minutes}min{seconds}s"
    return f"{seconds}s"

def run_cmd(cmd, cwd: Path):
    try:
        proc = subprocess.run(cmd, cwd=str(cwd),
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              timeout=1200,
                              encoding='utf-8',
                              errors='replace',
                              text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        timeout_stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
        timeout_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
        return -999, timeout_stdout, f"TIMEOUT: Process killed after 1200 seconds\n{timeout_stderr}"
    except Exception as e:
        return -998, "", f"ERROR: {str(e)}"

def clean_compiler_coverage(target_compiler):
    """Clean compiler coverage data to ensure each experiment starts from zero"""
    try:
        # Get project root directory
        script_dir = Path(__file__).parent
        project_root = script_dir
        
        # Determine coverage directory based on compiler type
        if target_compiler in ["gcc", "g++"]:
            coverage_dir = project_root / "target" / target_compiler / "gcc-coverage-build" / "gcc"
            
            if coverage_dir.exists():
                # Use find command to clean .gcda and coverage.info files
                cmd = ["find", str(coverage_dir), "(", "-name", "*.gcda", "-o", "-name", "coverage.info", ")", "-type", "f", "-delete"]
                returncode, stdout, stderr = run_cmd(cmd, project_root)
                
                if returncode == 0:
                    print(f"[INFO] Cleaned {target_compiler} compiler coverage data: {coverage_dir}")
                else:
                    print(f"[WARN] Failed to clean {target_compiler} compiler coverage data: {stderr}")
            else:
                print(f"[WARN] {target_compiler} compiler coverage directory does not exist: {coverage_dir}")
        else:
            print(f"[INFO] Compiler {target_compiler} does not need coverage data cleanup")
            
    except Exception as e:
        print(f"[ERROR] Error occurred while cleaning compiler coverage data: {e}")

def generate_fuzzing_inputs(generator, target_compiler, codes_dir, gen_log, calls_budget, batch_size):
    """
    Args:
        generator: LLM generator instance
        target_compiler: Target compiler name
        codes_dir: Code output directory
        gen_log: Generation log file path
        calls_budget: Maximum number of LLM calls
        batch_size: Batch size for generation

    Returns:
        tuple: (llm_calls, valid_codes)
    """
    # Get target language and file extension
    compiler_info = COMPILER_INFO.get(target_compiler, {"ext": ".java", "lang": "java"})
    target_language = compiler_info["lang"]
    file_extension = compiler_info["ext"]

    llm_calls = 0
    valid = 0
    start = time.time()
    append_log(gen_log, f"llm_calls,valid,filename")
    
    print(f"\n[INFO] 开始批量生成，batch_size={batch_size}")
    print(f"[INFO] 调用次数预算: {calls_budget} calls")
    
    pbar_gen = tqdm(total=calls_budget, unit="call", desc=f"Generate {target_compiler}", leave=True)
    
    # 批处理生成逻辑
    while llm_calls < calls_budget:
        # 计算本批次实际需要生成的数量
        remaining = calls_budget - llm_calls
        current_batch_size = min(batch_size, remaining)
        
        batch_prompts = []
        for _ in range(current_batch_size):
            prompt = f"Generate a {target_language} code snippet that can trigger compiler crash. Strictly use the format of:<think>the content of the thinking</think><code>content code</code>"
            batch_prompts.append(prompt)
        
        if not batch_prompts:
            break
        
        # 批量生成
        batch_start = time.time()
        codes = generator.generate_batch(batch_prompts)
        batch_time = time.time() - batch_start
        
        # 计算速度
        tokens_per_sec = (len(batch_prompts) * generator.max_new_tokens) / batch_time if batch_time > 0 else 0
        
        # 处理生成结果
        for code in codes:
            llm_calls += 1
            pbar_gen.update(1)
            
            if code and code.strip():
                valid += 1
                fname = f"case_{valid}{file_extension}"
                (codes_dir / fname).write_text(code, encoding="utf-8")
                append_log(gen_log, f"{llm_calls},{valid},{fname}")
        
        # 更新进度条信息
        pbar_gen.set_postfix({
            'valid': f"{valid}/{llm_calls}",
            'rate': f"{valid/llm_calls:.1%}" if llm_calls > 0 else "0%",
            'batch_time': f"{batch_time:.2f}s",
            'speed': f"{len(batch_prompts)/batch_time:.1f} req/s" if batch_time > 0 else "N/A"
        })

    pbar_gen.close()
    
    elapsed = time.time() - start
    print(f"\n[INFO] 生成完成！")
    print(f"[INFO] 总调用次数: {llm_calls}")
    print(f"[INFO] 有效代码数: {valid}")
    print(f"[INFO] 有效率: {valid/llm_calls:.2%}" if llm_calls > 0 else "[INFO] 有效率: 0%")
    print(f"[INFO] 总耗时: {format_hms(elapsed)}")
    print(f"[INFO] 平均速度: {llm_calls/elapsed:.2f} calls/秒" if elapsed > 0 else "[INFO] 平均速度: N/A")
    
    append_log(gen_log, f"Generated calls: {llm_calls},valid codes: {valid}")
    append_log(gen_log, f"valid rate: {valid/llm_calls:.2%}" if llm_calls > 0 else "valid rate: 0%")
    append_log(gen_log, f"Total time: {format_hms(elapsed)}")
    append_log(gen_log, f"Average speed: {llm_calls/elapsed:.2f} calls/s" if elapsed > 0 else "Average speed: N/A")

    return llm_calls, valid


def compile_source_files(target_compiler, work_dir, crashes_dir, comp_log, batch_id, batch_files):
    """
    Compile source files module
    
    Args:
        target_compiler: Target compiler name
        work_dir: Working directory path
        crashes_dir: Crash files directory
        comp_log: Compilation log file path
        batch_id: Batch ID
        batch_files: Source file list (all files in this batch)
    
    Returns:
        bool: Whether all files compiled successfully
    """
    print(f"\n=== Batch {batch_id}: {len(batch_files)} cases ===")
    
    # Get compilation script path
    compile_script = Path(__file__).parent / "target" / target_compiler / "compiler.sh"
    
    if not compile_script.exists():
        print(f"Error: Compile script not found: {compile_script}")
        return False
    
    # Ensure script has execute permission
    if not compile_script.stat().st_mode & 0o111:
        print(f"Warning: Compile script lacks execute permission: {compile_script}")
    
    all_success = True
    
    for src in batch_files:
        print(f"Compiling: {src.name}")
        
        # Call compilation script
        # Input: work_dir, complete source file path
        # Output: compilation result (return_code, stdout, stderr)
        ret, out, err = run_cmd(
            ["bash", str(compile_script), str(work_dir), str(src)],  # Pass complete path
            cwd=compile_script.parent  # Execute in script directory
        )
        
        # Check if crash.sh script exists for this compiler
        crash_script = Path(__file__).parent / "target" / target_compiler / "crash.sh"
        
        if crash_script.exists():
            
            # Call crash.sh with parameters: return_code, stdout, stderr
            crash_ret, crash_out, crash_err = run_cmd(
                ["bash", str(crash_script), str(ret), out, err],
                cwd=crash_script.parent
            )
            
            if crash_ret != 0:
                # crash.sh detected a crash or error
                all_success = False
                log_entry = [
                    f"--- CRASH batch {batch_id}: {src.name} (exit {ret}) ---",
                    "=== STDOUT ===",
                    out.strip(),
                    "=== STDERR ===",
                    err.strip(),
                    "=== CRASH DETECTION OUTPUT ===",
                    crash_out.strip(),
                    "==================================================================",
                    ""
                ]
                append_log(comp_log, "\n".join(log_entry))
                
                # Save crash source code and log (fuzz.py handles this)
                crash_src_file = crashes_dir / f"crash_{batch_id}_{src.name}"
                shutil.copy2(src, crash_src_file)
                
                # Save crash log information
                crash_log_file = crashes_dir / f"crash_{batch_id}_{src.stem}.log"
                crash_info = [
                    f"Compilation failed for: {src.name}",
                    f"Batch ID: {batch_id}",
                    f"Exit code: {ret}",
                    f"Time: -",
                    "",
                    "=== STDOUT ===",
                    out.strip(),
                    "",
                    "=== STDERR ===",
                    err.strip(),
                    "",
                    "=== CRASH DETECTION OUTPUT ===",
                    crash_out.strip()
                ]
                crash_log_file.write_text("\n".join(crash_info), encoding="utf-8")
                
                print(f"Crash detected by script: {src.name}")
                print(f"Crash case saved: {crash_src_file}")
                print(f"Crash log saved: {crash_log_file}")
            else:
                # No crash detected, compilation successful
                append_log(comp_log, f"{batch_id},{src.name},OK,{out.strip()}")
        else:
            # No crash.sh script, use original logic based on return code
            if ret != 0:
                # Compilation failed, record detailed information
                all_success = False
                log_entry = [
                    f"--- FAIL batch {batch_id}: {src.name} (exit {ret}) ---",
                    "=== STDOUT ===",
                    out.strip(),
                    "=== STDERR ===",
                    err.strip(),
                    "==================================================================",
                    ""
                ]
                append_log(comp_log, "\n".join(log_entry))
                
                # Save source code
                crash_src_file = crashes_dir / f"crash_{batch_id}_{src.name}"
                shutil.copy2(src, crash_src_file)
                
                # Save compilation error information
                crash_log_file = crashes_dir / f"crash_{batch_id}_{src.stem}.log"
                crash_info = [
                    f"Compilation failed for: {src.name}",
                    f"Batch ID: {batch_id}",
                    f"Exit code: {ret}",
                    f"Time: -",
                    "",
                    "=== STDOUT ===",
                    out.strip(),
                    "",
                    "=== STDERR ===",
                    err.strip()
                ]
                crash_log_file.write_text("\n".join(crash_info), encoding="utf-8")
                
                print(f"Crash case saved: {crash_src_file}")
                print(f"Crash log saved: {crash_log_file}")
            else:
                # Compilation successful, record status
                append_log(comp_log, f"{batch_id},{src.name},OK")
    
    return all_success


def collect_coverage(target_compiler, work_dir, cov_log, batch_id, coverage_interval_seconds):
    """
    Collect code coverage
    
    Args:
        target_compiler: Target compiler name
        work_dir: Working directory path
        codes_dir: Code directory
        cov_log: Coverage log file path
        batch_id: Batch ID
        coverage_interval_seconds: Time interval per batch (seconds)
    
    Returns:
        bool: Whether coverage collection was successful
    """
    # Get coverage collection script path
    coverage_script = Path(__file__).parent / "target" / target_compiler / "coverage.sh"
    
    if not coverage_script.exists():
        print(f"Error: Coverage script not found: {coverage_script}")
        return False
    
    # Ensure script has execute permission
    if not coverage_script.stat().st_mode & 0o111:
        print(f"Warning: Coverage script lacks execute permission: {coverage_script}")
    
    # Call coverage collection script
    # Input: work_dir parameter
    # Output: coverage data (return_code, stdout, stderr)
    ret, out, err = run_cmd(
        ["bash", str(coverage_script), str(work_dir)],
        cwd=coverage_script.parent  # Execute in script directory
    )
    
    if ret == 0:
        try:
            # Parse coverage output
            covered = out.strip()
            # Use batch number * coverage interval seconds as time record (HH:MM:SS format)
            # but not exceeding time budget
            elapsed_seconds = batch_id * int(coverage_interval_seconds)
            elapsed_hms = format_hms(elapsed_seconds)
            append_log(cov_log, f"{elapsed_hms},{covered}")
            
            return True
            
        except ValueError as e:
            print(f"Error parsing coverage output: {e}")
            print(f"Raw output: {out.strip()}")
            return False
    else:
        print(f"[!] Coverage failed on batch {batch_id}:\n{err}")
        return False

def main():
    # Parsing config.yaml
    p = argparse.ArgumentParser()
    p.add_argument("config", help="Path to config.yaml")
    p.add_argument("--gpu", type=str, default="0", help="GPU devices to use, e.g., '0' or '0,1,2,3'")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    target_compiler = cfg["target"]
    work_dir        = Path(cfg["work_dir"]).expanduser().resolve()
    model_name      = cfg["model_name"]
    temperature     = cfg["temperature"]
    max_length      = cfg.get("max_length", 512)  # 保持向后兼容
    max_new_tokens  = cfg.get("max_new_tokens", max_length)  # 优先使用 max_new_tokens
    batch_size      = cfg["batch_size"]
    calls_budget    = cfg["calls_budget"]  # 使用调用次数预算
    coverage_interval_seconds = int(cfg.get("coverage_interval_seconds", 3600))
    
    # Parse GPU parameters
    gpu_devices = [int(x.strip()) for x in args.gpu.split(",")]

    # Initialize the working directory
    if work_dir.exists():
        print(" Error: work_dir exists")
        return 0

    codes_dir    = work_dir / "codes"
    coverage_dir = work_dir / "coverage"
    crashes_dir = work_dir / "crashes"
    logs_dir     = work_dir / "logs"
    gen_log      = logs_dir / "generation.log"
    comp_log     = logs_dir / "compiler.log"
    cov_log      = logs_dir / "coverage.log"

    # Create all required directories
    for d in (codes_dir, coverage_dir, crashes_dir, logs_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    
    # Create log files
    for logf in (gen_log, comp_log, cov_log):
        logf.parent.mkdir(parents=True, exist_ok=True)
        logf.write_text("", encoding="utf-8")

    # Clean compiler coverage data (if gcc/g++ compiler)
    clean_compiler_coverage(target_compiler)

    print(f"\n{'='*60}")
    print(f"配置信息:")
    print(f"  - 目标编译器: {target_compiler}")
    print(f"  - 模型: {model_name}")
    print(f"  - GPU设备: {gpu_devices}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Max New Tokens: {max_new_tokens}")
    print(f"  - Temperature: {temperature}")
    print(f"  - 调用次数预算: {calls_budget} calls")
    print(f"  - 覆盖率收集间隔: 每 {coverage_interval_seconds} 秒")
    print(f"{'='*60}\n")

    # fuzzing inputs generator (使用优化版)
    generator = LLMGenerator(
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,  # 使用 max_new_tokens
        batch_size=batch_size,
        gpu_devices=gpu_devices
    )

    # Record experiment start time
    experiment_start_time = time.time()

    # 使用优化版批处理生成
    llm_calls, valid = generate_fuzzing_inputs(
        generator, 
        target_compiler, 
        codes_dir, 
        gen_log, 
        calls_budget,  # 传递调用次数预算
        batch_size
    )

    # 按照文件生成时间分批（每 coverage_interval_seconds 秒的文件为一批）
    file_extension = COMPILER_INFO.get(target_compiler, {"ext": ".java"})["ext"]
    files = sorted(codes_dir.glob(f"case_*{file_extension}"))

    # 按文件生成时间分批：每 coverage_interval_seconds 秒的文件为一批
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    batches_dict = {}
    for f in files:
        elapsed = int(max(0, f.stat().st_mtime - experiment_start_time))
        batch_key = elapsed // coverage_interval_seconds
        batches_dict.setdefault(batch_key, []).append(f)
    batches = [batches_dict[k] for k in sorted(batches_dict.keys())]

    pbar_cc = tqdm(total=len(batches), unit="batch", desc="Compile & Cover", leave=True)
    
    append_log(cov_log, f"run_time, coverage")

    for batch_id, batch_files in enumerate(batches, start=1):
        # Compile this batch
        compile_source_files(target_compiler, work_dir, crashes_dir, comp_log, batch_id, batch_files)

        # Collect coverage once after each batch
        collect_coverage(target_compiler, work_dir, cov_log, batch_id, coverage_interval_seconds)

        pbar_cc.update(1)

    pbar_cc.close()
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print("="*60)
    print(f"LLM调用次数: {llm_calls}")
    print(f"有效代码数: {valid}")
    print(f"有效率: {valid/llm_calls:.2%}" if llm_calls > 0 else "有效率: 0%")
    print(f"\n日志文件:")
    print(f"  - 生成日志: {gen_log}")
    print(f"  - 编译日志: {comp_log}")
    print(f"  - 崩溃目录: {crashes_dir}")
    print(f"  - 覆盖率日志: {cov_log}")
    print("="*60)

if __name__ == "__main__":
    main()

