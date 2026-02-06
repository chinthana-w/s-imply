import re
import pandas as pd
import time

def main():
    log_path = "benchmark_vanilla_iscas85_parallel.log"
    output_md = "vanilla_iscas85_results.md"
    
    results = []
    
    # Regex to parse: "Finished c17.bench: FC=100.00%, Time=0.00s, Avg=0.07ms"
    # Actually, the log format is:
    # "Finished <bench>: FC=<val>%, Time=<val>s, Avg=<val>ms"
    
    pattern = re.compile(r"Finished (.+?): FC=(.+?)%, Time=(.+?)s, Avg=(.+?)ms")
    
    try:
        with open(log_path, "r") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    # We don't have total fault counts in this log line, 
                    # but we can at least report what we see.
                    # To get total faults, we might need to parse the earlier lines or just skip text.
                    # The parallel script return dict has it, but log print doesn't showing it explicitly 
                    # except implied by % and time.
                    # However, we can just list the parsed values for now.
                    circuit = match.group(1)
                    fc = float(match.group(2))
                    time_s = float(match.group(3))
                    avg_ms = float(match.group(4))
                    
                    results.append({
                        "Circuit": circuit,
                        "Coverage (%)": fc,
                        "Time (s)": time_s,
                        "Avg Time/Fault (ms)": avg_ms
                    })
    except FileNotFoundError:
        print(f"Log file {log_path} not found.")
        return

    if not results:
        print("No results found in log yet.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("Circuit")
    
    # Format
    df_disp = df.copy()
    df_disp['Coverage (%)'] = df_disp['Coverage (%)'].map('{:,.2f}'.format)
    df_disp['Time (s)'] = df_disp['Time (s)'].map('{:,.2f}'.format)
    df_disp['Avg Time/Fault (ms)'] = df_disp['Avg Time/Fault (ms)'].map('{:,.2f}'.format)
    
    md_table = df_disp.to_markdown(index=False)
    
    report = f"""# ISCAS85 Vanilla PODEM Benchmark Results (Partial)
**Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}**
**Source: {log_path}**

{md_table}
"""
    
    with open(output_md, "w") as f:
        f.write(report)
        
    print(f"Generated {output_md} from log.")
    print(df_disp)

if __name__ == "__main__":
    main()
