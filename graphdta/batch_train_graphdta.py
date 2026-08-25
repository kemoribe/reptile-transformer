"""
批量训练 GraphDTA 模型脚本

依次运行多个模型训练任务，支持断点续训和日志记录。

使用方法:
    python batch_train_graphdta.py                  # 运行所有任务
    python batch_train_graphdta.py --resume          # 从断点继续
    python batch_train_graphdta.py --start-from 1    # 从第2个任务开始
    python batch_train_graphdta.py --only 0          # 只运行第0个任务
    python batch_train_graphdta.py --skip            # 跳过已完成任务

任务列表:
    0: chembl + GATNet        (training.py 0 1)
    1: chembl + GAT_GCN       (training.py 0 2)
    2: chembl + GCNNet        (training.py 0 3)
    3: davis  + GINConvNet    (training.py 1 0)
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "batch_training_state.json"
LOG_DIR = BASE_DIR / "batch_training_logs"

TASKS = [
    {
        "name": "chembl_GATNet",
        "dataset_idx": 0,
        "model_idx": 1,
        "description": "chembl + GATNet",
    },
    {
        "name": "chembl_GAT_GCN",
        "dataset_idx": 0,
        "model_idx": 2,
        "description": "chembl + GAT_GCN",
    },
    {
        "name": "chembl_GCNNet",
        "dataset_idx": 0,
        "model_idx": 3,
        "description": "chembl + GCNNet",
    },
    {
        "name": "davis_GINConvNet",
        "dataset_idx": 1,
        "model_idx": 0,
        "description": "davis + GINConvNet",
    },
]


def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"completed_tasks": [], "current_task": None, "history": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_checkpoint(task):
    """检查任务是否有可续训的 checkpoint"""
    checkpoint_file = BASE_DIR / f"checkpoint_GATNet_chembl.pt"
    # 根据任务动态生成文件名
    from models.gat import GATNet
    from models.gat_gcn import GAT_GCN
    from models.gcn import GCNNet
    from models.ginconv import GINConvNet
    
    model_names = ['GINConvNet', 'GATNet', 'GAT_GCN', 'GCNNet']
    dataset_names = ['chembl', 'davis', 'kiba']
    
    model_name = model_names[task['model_idx']]
    dataset_name = dataset_names[task['dataset_idx']]
    
    checkpoint_file = BASE_DIR / f"checkpoint_{model_name}_{dataset_name}.pt"
    return checkpoint_file.exists()


def run_task(task_idx, task, skip_completed=False):
    script_path = BASE_DIR / "training.py"
    log_file = LOG_DIR / f"{task['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    cmd = [
        sys.executable,
        str(script_path),
        str(task['dataset_idx']),
        str(task['model_idx']),
        "0",  # GPU ID
    ]
    
    # 检查是否有 checkpoint 可以续训
    has_checkpoint = check_checkpoint(task)
    if has_checkpoint:
        cmd.append("--resume")
        print(f"  [Resume] Found checkpoint, will resume training")
    
    print(f"\n{'='*70}")
    print(f"Task {task_idx}: {task['description']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    with open(log_file, "w") as log_f:
        log_f.write(f"Task: {task['description']}\n")
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Start time: {datetime.now()}\n\n")
        
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        
        process.wait()
    
    elapsed = time.time() - start_time
    status = "success" if process.returncode == 0 else f"failed (exit code: {process.returncode})"
    
    print(f"\n{'='*70}")
    print(f"Task {task_idx} finished: {status}, time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}\n")
    
    return process.returncode == 0, elapsed


def load_result_metrics(task):
    """读取训练结果 JSON 文件，返回指标字典"""
    from models.gat import GATNet
    from models.gat_gcn import GAT_GCN
    from models.gcn import GCNNet
    from models.ginconv import GINConvNet
    
    model_names = ['GINConvNet', 'GATNet', 'GAT_GCN', 'GCNNet']
    dataset_names = ['chembl', 'davis', 'kiba']
    
    model_name = model_names[task['model_idx']]
    dataset_name = dataset_names[task['dataset_idx']]
    
    result_file = BASE_DIR / f"result_{model_name}_{dataset_name}.json"
    
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            return data.get("test_metrics", None)
        except:
            pass
    return None


def print_metrics_table(results, tasks):
    """打印指标汇总表格 (包含 AUPR)"""
    metric_keys = [
        ('RMSE', '.4f'), ('MSE', '.4f'), ('MAE', '.4f'), ('R2', '.4f'),
        ('Pearson', '.4f'), ('Spearman', '.4f'), ('CI', '.4f'),
        ('EF@1%', '.2f'), ('EF@5%', '.2f'), ('EF@10%', '.2f'),
        ('ECE', '.4f'), ('AUPR', '.4f'),
    ]
    
    # 收集所有已完成任务的指标
    all_metrics = []
    for r in results:
        if r.get("status") == "success" and "idx" in r:
            task = tasks[r["idx"]]
            metrics = load_result_metrics(task)
            if metrics:
                row = {"task": task["description"], "best_epoch": metrics.get("best_epoch", "-")}
                for key, fmt in metric_keys:
                    row[key] = metrics.get(key, None)
                all_metrics.append(row)
    
    if not all_metrics:
        return
    
    print(f"\n{'='*120}")
    print("训练结果汇总表 (包含 AUPR)")
    print(f"{'='*120}")
    
    # 表头
    header = f"{'Task':<25} {'Epoch':>5}"
    for key, _ in metric_keys:
        header += f" {key:>10}"
    print(header)
    print("-" * len(header))
    
    # 数据行
    for row in all_metrics:
        line = f"{row['task']:<25} {str(row['best_epoch']):>5}"
        for key, fmt in metric_keys:
            val = row.get(key)
            if val is not None:
                line += f" {val:{fmt}}"
            else:
                line += f" {'N/A':>10}"
        print(line)
    
    print(f"{'='*120}")
    
    # 保存汇总结果
    summary_file = BASE_DIR / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "completed": len(all_metrics),
            "results": all_metrics,
        }, f, indent=2)
    print(f"\n汇总结果已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="批量训练 GraphDTA 模型")
    parser.add_argument("--resume", action="store_true", help="从断点继续训练（跳过已完成任务）")
    parser.add_argument("--start-from", type=int, default=None, help="从指定索引开始")
    parser.add_argument("--only", type=int, default=None, help="只运行指定索引的任务")
    parser.add_argument("--list", action="store_true", help="列出所有任务")
    parser.add_argument("--skip", action="store_true", help="跳过已完成的任务")
    parser.add_argument("--reset", action="store_true", help="重置训练状态")
    parser.add_argument("--summary", action="store_true", help="仅打印结果汇总（不运行训练）")
    args = parser.parse_args()
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        print("\n任务列表:")
        for i, task in enumerate(TASKS):
            print(f"  {i}: {task['description']}")
        print()
        return
    
    if args.summary:
        print("\n读取已有训练结果...")
        dummy_results = []
        for i, task in enumerate(TASKS):
            metrics = load_result_metrics(task)
            if metrics:
                dummy_results.append({"idx": i, "status": "success", "time": 0})
        print_metrics_table(dummy_results, TASKS)
        return
    
    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            print("训练状态已重置")
        return
    
    state = load_state()
    completed = set(state.get("completed_tasks", []))
    
    print(f"\n当前状态: 已完成 {len(completed)}/{len(TASKS)} 个任务")
    if completed:
        print(f"  已完成任务: {sorted(completed)}")
    
    skip_completed = args.resume or args.skip
    
    start_idx = args.start_from if args.start_from is not None else 0
    end_idx = args.only + 1 if args.only is not None else len(TASKS)
    
    total_start = time.time()
    results = []
    
    for i in range(start_idx, end_idx):
        task = TASKS[i]
        
        if skip_completed and i in completed:
            print(f"\n[Skip] Task {i}: {task['description']} (already completed)")
            results.append({"idx": i, "status": "skipped", "time": 0})
            continue
        
        state["current_task"] = i
        save_state(state)
        
        success, elapsed = run_task(i, task, skip_completed=skip_completed)
        
        result = {
            "idx": i,
            "name": task["name"],
            "description": task["description"],
            "status": "success" if success else "failed",
            "time": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        results.append(result)
        
        if success:
            state["completed_tasks"].append(i)
            state["history"].append(result)
            save_state(state)
        else:
            print(f"\n[Error] Task {i} failed! Check log: {LOG_DIR}")
            print("Use --resume to continue from where it left off")
            break
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print("批量训练完成!")
    print(f"{'='*70}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"\n任务状态汇总:")
    for r in results:
        if "idx" in r:
            task_desc = TASKS[r["idx"]]["description"]
            status_icon = "✓" if r["status"] == "success" else ("⏭" if r["status"] == "skipped" else "✗")
            time_str = f"{r['time']:.1f}s" if r["status"] != "skipped" else "-"
            print(f"  {status_icon} Task {r['idx']}: {task_desc} [{r['status']}] ({time_str})")
        else:
            print(f"  ? {r}")
    
    # 打印指标汇总表 (包含 AUPR)
    print_metrics_table(results, TASKS)
    
    print(f"\n日志目录: {LOG_DIR}")
    print(f"状态文件: {STATE_FILE}")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    
    if failed_count == 0:
        print(f"\n🎉 成功完成 {success_count} 个任务!")
        print("使用 --summary 查看完整指标汇总")
    else:
        print(f"\n⚠️  {failed_count} 个任务失败")
        print("使用 --resume 重新运行未完成的任务")


if __name__ == "__main__":
    main()