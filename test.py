import json
import os
import platform
try:
    import resource
except ImportError:
    resource = None

# === Load cgroup limits from Docker inspect ===
# Make sure docker_config.json is present in the same directory
cfg = json.load(open("docker_config.json"))[0]["HostConfig"]
mem_limit = cfg.get("Memory", 0)          # bytes
cpu_quota = cfg.get("CpuQuota", 0)        # microseconds per period
cpu_period = cfg.get("CpuPeriod", 0)      # microseconds

# Enforce memory limit (address space) if supported
if resource and platform.system() != "Windows":
    if mem_limit > 0:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
else:
    print("⚠️ Resource limits disabled on this platform")

# Enforce CPU time limit if supported
if resource and platform.system() != "Windows" and cpu_quota > 0 and cpu_period > 0:    
    cores = cpu_quota / cpu_period
    cpu_seconds = int(60 * cores)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
else:
    print("⚠️ CPU limits disabled on this platform")

# --- Rest of test harness follows ---
import asyncio
import time
import subprocess
from pathlib import Path
from importlib import import_module
from google.genai.errors import ClientError
import os
import csv

# --- CONFIGURE: module names of your three clients ---
CLIENT_MODULES = ["client", "client2", "client3"]
SERVER_SCRIPT = "server.py"
RUNS = 10

TESTS = [
    {"name":"no_tool_hello",
     "prompt":"hi",
     "check": lambda out: True},

    {"name":"list_py",
     "prompt":"List all .py files in .",
     "check": lambda out: all(str(p) in out for p in Path('.').glob('*.py'))},

    {"name":"append_foo",
     "prompt":"Append FOO to test.txt",
     "check_file":"test.txt",
     "check": lambda _: Path("test.txt").read_text().endswith("FOO")},

    {"name":"read_test",
     "prompt":"Read test.txt",
     "check": lambda out: "FOO" in out},

    {"name":"create_add_script",
     "prompt":"Create a Python file named 'add.py' that takes two numbers as input and prints their sum.",
     "check_file":"add.py",
     "check": lambda _: (
         Path("add.py").exists() and (
             # detect sys.argv usage
             ("sys.argv" in Path("add.py").read_text() and subprocess.run(
                 ["python", "add.py", "3", "5"], capture_output=True, text=True
             ).stdout.strip() == "8") or
             # else interactive input
             subprocess.run(
                 ["python", "add.py"], input="3\n5\n",
                 capture_output=True, text=True
             ).stdout.strip() == "8"
         )
     )},
]

async def run_single(client, test):
    if test.get("check_file"):
        try:
            os.remove(test["check_file"])
        except FileNotFoundError:
            pass

    start = time.perf_counter()
    try:
        raw = await client.process_query(test["prompt"])
        elapsed = time.perf_counter() - start

        if isinstance(raw, dict):
            text  = raw.get("text", "")
            usage = raw.get("usage") or raw.get("usage_metadata", {})
        else:
            text, usage = raw, {}

        tokens = "-"
        if usage.get("prompt_tokens") is not None:
            tokens = usage["prompt_tokens"] + usage["completion_tokens"]
        elif usage.get("prompt_token_count") is not None:
            tokens = usage["prompt_token_count"] + usage["response_token_count"]

        if "check_file" in test:
            success = test["check"](None)
        else:
            success = test["check"](text)

        return {
            "test":    test["name"],
            "success": bool(success),
            "time_s":  elapsed,
            "tokens":  tokens,
            "output":  (text[:100] + "…") if len(text)>100 else text
        }

    except ClientError as e:
        return {"test": test["name"], "success": False, "time_s": None, "tokens": "-", "output": f"ERROR: {e}"}
    except Exception as e:
        return {"test": test["name"], "success": False, "time_s": None, "tokens": "-", "output": f"ERROR: {e}"}

async def run_client_tests(module_name):
    mod = import_module(module_name)
    ClientCls = (getattr(mod, "MCPClient", None)
                 or getattr(mod, "GeminiClient", None)
                 or getattr(mod, "SimpleChatClient", None))
    client = ClientCls()
    if hasattr(client, "connect_to_server"):
        await client.connect_to_server(SERVER_SCRIPT)

    results = []
    for test in TESTS:
        res = await run_single(client, test)
        res["client"] = module_name
        results.append(res)

    if hasattr(client, "cleanup"):
        await client.cleanup()
    return results

async def main():
    all_results = []

    for run_id in range(1, RUNS + 1):
        for mod in CLIENT_MODULES:
            res_list = await run_client_tests(mod)
            for r in res_list:
                r["run"] = run_id
                all_results.append(r)

    # Print markdown-style result table
    print("| Run | Client | Test | Success | Time (s) | Tokens | Output |")
    print("|----:|:-------|:-----|:-------:|:--------:|:------:|:-------|")
    for r in all_results:
        time_s = f"{r['time_s']:.3f}" if isinstance(r['time_s'], float) else "-"
        print(f"| {r['run']} | {r['client']} | {r['test']} | {'✅' if r['success'] else '❌'} | {time_s} | {r['tokens']} | {r['output']} |")

    # Export results to CSV
    csv_path = "results.csv"
    try:
        with open(csv_path, "w", newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=["run", "client", "test", "success", "time_s", "tokens", "output"])
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    k: ("TRUE" if r[k] is True else "FALSE" if k == "success" else r[k])
                    for k in ["run", "client", "test", "success", "time_s", "tokens", "output"]
                })
        print(f"\nExported results to {csv_path}")
    except PermissionError as e:
        print(f"\nCould not write CSV: {e}")
        
        
    
    
    # for mod in CLIENT_MODULES:
    #     all_results.extend(await run_client_tests(mod))

    # # Print markdown table
    # print("| Client   | Test                 | Success | Time (s)  | Tokens | Output |")
    # print("|---------:|:---------------------|:-------:|:---------:|:------:|:-------|")
    # for r in all_results:
    #     time_s = f"{r['time_s']:.3f}" if isinstance(r['time_s'], float) else "-"
    #     print(f"| {r['client']} | {r['test']} | {'✅' if r['success'] else '❌'} | {time_s} | {r['tokens']} | {r['output']} |")

    # # Export to CSV (safe)
    # csv_path = "results.csv"
    # try:
    #     with open(csv_path, "w", newline="", encoding="utf-8") as cf:
    #         writer = csv.DictWriter(cf, fieldnames=["client","test","success","time_s","tokens","output"])
    #         writer.writeheader()
    #         for r in all_results:
    #             writer.writerow({
    #                 "client":  r["client"],
    #                 "test":    r["test"],
    #                 "success": "TRUE" if r["success"] else "FALSE",
    #                 "time_s":  "" if r["time_s"] is None else f"{r['time_s']:.3f}",
    #                 "tokens":  r["tokens"],
    #                 "output":  r["output"].replace("\n","\\n")
    #             })
    #     print(f"\nExported results to {csv_path}")
    # except PermissionError as e:
    #     print(f"\nCould not write CSV: {e}")

if __name__ == "__main__":
    asyncio.run(main())
