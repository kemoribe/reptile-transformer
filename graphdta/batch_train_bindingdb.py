"""批量训练 BindingDB 数据集的 4 个 GraphDTA 模型"""
import os, sys, json, time, subprocess, argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "batch_bindingdb_state.json"
LOG_DIR = BASE_DIR / "batch_bindingdb_logs"

TASKS = [
    {"name": "bindingdb_GINConvNet", "dataset_idx": 3, "model_idx": 0},
    {"name": "bindingdb_GATNet",     "dataset_idx": 3, "model_idx": 1},
    {"name": "bindingdb_GAT_GCN",    "dataset_idx": 3, "model_idx": 2},
    {"name": "bindingdb_GCNNet",     "dataset_idx": 3, "model_idx": 3},
]

MODEL_NAMES = ['GINConvNet', 'GATNet', 'GAT_GCN', 'GCNNet']
DATASET_NAMES = ['chembl', 'davis', 'kiba', 'bindingdb']

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {"completed_tasks": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def check_checkpoint(task):
    model_name = MODEL_NAMES[task['model_idx']]
    dataset_name = DATASET_NAMES[task['dataset_idx']]
    return (BASE_DIR / f"checkpoint_{model_name}_{dataset_name}.pt").exists()

def run_task(task_idx, task):
    script_path = BASE_DIR / "training.py"
    model_name = MODEL_NAMES[task['model_idx']]
    dataset_name = DATASET_NAMES[task['dataset_idx']]
    log_file = LOG_DIR / f"{task['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    cmd = [sys.executable, str(script_path), str(task['dataset_idx']), str(task['model_idx']), "0"]

    if check_checkpoint(task):
        cmd.append("--resume")
        print(f"  [Resume] Found checkpoint")

    print(f"\n{'='*70}")
    print(f"Task {task_idx}: {dataset_name} + {model_name}")
    print(f"{'='*70}\n")

    start_time = time.time()
    with open(log_file, "w") as log_f:
        log_f.write(f"Task: {dataset_name} + {model_name}\n")
        process = subprocess.Popen(
            cmd, cwd=str(BASE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        process.wait()

    elapsed = time.time() - start_time
    status = "success" if process.returncode == 0 else f"failed (exit {process.returncode})"
    print(f"\nTask {task_idx}: {status}, time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    return process.returncode == 0, elapsed

def load_result_metrics(task):
    model_name = MODEL_NAMES[task['model_idx']]
    dataset_name = DATASET_NAMES[task['dataset_idx']]
    result_file = BASE_DIR / f"result_{model_name}_{dataset_name}.json"
    if result_file.exists():
        try:
            with open(result_file, "r") as f:
                data = json.load(f)
            return data.get("test_metrics", None)
        except: pass
    return None

def print_metrics_table(results):
    metric_keys = [
        ('RMSE', '.4f'), ('MSE', '.4f'), ('MAE', '.4f'), ('R2', '.4f'),
        ('Pearson', '.4f'), ('Spearman', '.4f'), ('CI', '.4f'),
        ('EF@1%', '.2f'), ('EF@5%', '.2f'), ('EF@10%', '.2f'),
        ('ECE', '.4f'), ('AUPR', '.4f'),
    ]
    all_metrics = []
    for r in results:
        if r.get("status") == "success" and "idx" in r:
            task = TASKS[r["idx"]]
            metrics = load_result_metrics(task)
            if metrics:
                model_name = MODEL_NAMES[task['model_idx']]
                row = {"task": f"bindingdb + {model_name}", "best_epoch": metrics.get("best_epoch", "-")}
                for key, fmt in metric_keys:
                    row[key] = metrics.get(key, None)
                all_metrics.append(row)

    if not all_metrics:
        print("\nNo results yet.")
        return

    print(f"\n{'='*120}")
    print("BindingDB GraphDTA 训练结果汇总")
    print(f"{'='*120}")
    header = f"{'Task':<28} {'Epoch':>5}"
    for key, _ in metric_keys:
        header += f" {key:>10}"
    print(header)
    print("-" * len(header))
    for row in all_metrics:
        line = f"{row['task']:<28} {str(row['best_epoch']):>5}"
        for key, fmt in metric_keys:
            val = row.get(key)
            line += f" {val:{fmt}}" if val is not None else f" {'N/A':>10}"
        print(line)
    print(f"{'='*120}")

    summary_file = BASE_DIR / "bindingdb_training_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_metrics}, f, indent=2)
    print(f"\n汇总已保存: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="批量训练 BindingDB GraphDTA 模型")
    parser.add_argument("--resume", action="store_true", help="从断点继续")
    parser.add_argument("--start-from", type=int, default=None)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--skip", action="store_true", help="跳过已完成任务")
    parser.add_argument("--summary", action="store_true", help="仅打印结果")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        print("\n任务列表:")
        for i, t in enumerate(TASKS):
            print(f"  {i}: {DATASET_NAMES[t['dataset_idx']]} + {MODEL_NAMES[t['model_idx']]}")
        return

    if args.summary:
        dummy = []
        for i, t in enumerate(TASKS):
            if load_result_metrics(t):
                dummy.append({"idx": i, "status": "success"})
        print_metrics_table(dummy)
        return

    state = load_state()
    completed = set(state.get("completed_tasks", []))
    print(f"\n已完成 {len(completed)}/{len(TASKS)} 个任务: {sorted(completed) if completed else '无'}")

    skip_completed = args.resume or args.skip
    start_idx = args.start_from if args.start_from is not None else 0
    end_idx = args.only + 1 if args.only is not None else len(TASKS)

    results = []
    total_start = time.time()

    for i in range(start_idx, end_idx):
        task = TASKS[i]
        if skip_completed and i in completed:
            print(f"\n[Skip] Task {i}: {MODEL_NAMES[task['model_idx']]} (已完成)")
            results.append({"idx": i, "status": "skipped", "time": 0})
            continue

        state["current_task"] = i
        save_state(state)
        success, elapsed = run_task(i, task)
        results.append({"idx": i, "status": "success" if success else "failed", "time": elapsed})

        if success:
            state["completed_tasks"].append(i)
            save_state(state)
        else:
            print(f"\n[Error] Task {i} 失败! 使用 --resume 继续")
            break

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"批量训练完成! 总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    for r in results:
        if "idx" in r:
            m = MODEL_NAMES[TASKS[r["idx"]]["model_idx"]]
            icon = "✓" if r["status"] == "success" else ("⏭" if r["status"] == "skipped" else "✗")
            print(f"  {icon} Task {r['idx']}: bindingdb + {m} [{r['status']}]")
    print_metrics_table(results)

if __name__ == "__main__":
    main()
