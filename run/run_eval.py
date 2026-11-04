import sys
from evalscope.run import run_task
from evalscope.summarizer import Summarizer


def run_one(task_cfg: str):
    print(f"\n===== Running benchmark =====")
    print(f">> Config: {task_cfg}")

    # 1) run evaluation
    run_task(task_cfg=task_cfg)

    # 2) summarize results
    print(">> Collecting report with Summarizer ...")
    report_list = Summarizer.get_report_from_cfg(task_cfg)

    print(">> Report:")
    for r in report_list:
        print(r)

    return report_list


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_eval.py <config1.yaml> [config2.yaml ...]")
        sys.exit(1)

    cfgs = sys.argv[1:]
    all_reports = {}

    for cfg in cfgs:
        all_reports[cfg] = run_one(cfg)

    print("\n===== All benchmarks finished =====")
    for cfg, reports in all_reports.items():
        print(f"\n[{cfg}]")
        for r in reports:
            print(r)


if __name__ == "__main__":
    main()
