import argparse
import ast
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    exit(1)

class ModelSampler:
    def __init__(self, extra_keys_str: str = ""):
        self.all_env_keys: Dict[str, str] = {}
        if extra_keys_str:
            for pair in extra_keys_str.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    self.all_env_keys[k.strip()] = v.strip()

    def run(self, num_samples: int, api_weights_str: str, model_weights_str: str, 
            blocked_str: str, exclude_str: str, completed_str: str):
        
        model_cfg = ast.literal_eval(model_weights_str)
        spec_weights = ast.literal_eval(api_weights_str)
        partial_done = ast.literal_eval(completed_str)
        
        blocked_keys = set([k.strip() for k in blocked_str.split(',') if k.strip()])
        exclude_keys = set([k.strip() for k in exclude_str.split(',') if k.strip()])
        
        # 1. Key Discovery & Deduplication (Value-based)
        unique_keys_by_provider = {}
        all_unique_names = []
        for p in model_cfg.keys():
            unique_keys_by_provider[p] = []
            seen_vals = {}
            for k, v in self.all_env_keys.items():
                if p.lower() in k.lower():
                    if v not in seen_vals:
                        seen_vals[v] = k
                        unique_keys_by_provider[p].append(k)
                        all_unique_names.append(k)

        # 2. Initial Weight Plan
        sum_spec = sum(spec_weights.get(k, 0.0) for k in all_unique_names)
        unspec = [k for k in all_unique_names if k not in spec_weights]
        base_share = (max(0.0, 1.0 - sum_spec) / len(unspec)) if unspec else 0.0
        init_weights = {k: spec_weights.get(k, base_share) for k in all_unique_names}

        # 3. Start-of-Run Redistribution
        final_weights = {}
        start_blocked = [k for k in blocked_keys if k not in partial_done]
        harvest = sum(init_weights[k] for k in start_blocked)
        targets = [k for k in all_unique_names if k not in blocked_keys and k not in exclude_keys]
        bonus = harvest / len(targets) if targets else 0.0

        for k in all_unique_names:
            if k in start_blocked: continue
            final_weights[k] = init_weights[k] + (bonus if k not in exclude_keys else 0.0)

        # 4. Model Normalization
        norm_models = {p.lower(): {m: (w if w is not None else (max(0.0, 1.0 - sum(v for v in models.values() if v is not None)) / len([mk for mk, mv in models.items() if mv is None]))) 
                       for m, w in models.items()} for p, models in model_cfg.items()}

        # 5. Build Initial Queue
        queue = []
        key_totals = {} 
        for p_name, keys in unique_keys_by_provider.items():
            for k in keys:
                if k not in final_weights: continue
                for m_name, m_w in norm_models[p_name].items():
                    count = int(round(final_weights[k] * m_w * num_samples))
                    queue.append({"key": k, "model": m_name, "target": count, "provider": p_name})
                    key_totals[k] = key_totals.get(k, 0) + count

        # 6. Execute Simulation with Multi-TQDM
        print(f"\n🚀 Simulation: Processing {num_samples} samples across unique keys...\n")
        
        bars = {}
        active_keys = sorted(key_totals.keys())
        for i, k in enumerate(active_keys):
            # FIXED: Removed k[:15] to show full OLLAMA_API_KEY_X
            bars[k] = tqdm(total=key_totals[k], desc=f"🔑 {k}", position=i, leave=True)
        
        global_bar = tqdm(total=num_samples, desc="🌍 TOTAL PROGRESS", position=len(active_keys), unit="sample")

        results = {}
        while queue:
            task = queue.pop(0)
            k, m, target = task['key'], task['model'], task['target']
            limit = partial_done.get(k, {}).get(m, float('inf'))
            
            processed = 0
            for _ in range(target):
                if processed >= limit:
                    # MID-RUN FAIL Logic
                    rem = target - processed
                    failover_targets = [tk for tk in unique_keys_by_provider[task['provider']] 
                                        if tk not in blocked_keys and tk not in exclude_keys and tk != k]
                    
                    if failover_targets:
                        target_key = failover_targets[0]
                        # Move tasks to Target Key
                        queue.append({"key": target_key, "model": m, "target": rem, "provider": task['provider']})
                        
                        # Adjust Bar Totals dynamically
                        bars[target_key].total += rem
                        bars[target_key].refresh()
                        bars[k].total -= rem
                        bars[k].refresh()
                    break
                
                time.sleep(0.0005) 
                processed += 1
                bars[k].update(1)
                global_bar.update(1)
            
            results.setdefault(k, {}).setdefault(m, 0)
            results[k][m] += processed
        
        for b in bars.values(): b.close()
        global_bar.close()

        # 7. Final Report
        print("\n" * (len(active_keys) + 1)) 
        print(f"--- Final Distribution ---")
        for p_name in model_cfg.keys():
            print(f"\n📁 Provider: {p_name.upper()}")
            for k in sorted(all_unique_names):
                if p_name.lower() in k.lower() and k in results:
                    status = "❌" if k in blocked_keys else "🔑"
                    print(f"  {status} {k}:")
                    for m, count in results[k].items():
                        print(f"     - {m:20}: {count:>6} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--samples", type=int, default=5000)
    parser.add_argument("--model-weights", type=str, required=True)
    parser.add_argument("--api-weights", type=str, default="{}")
    parser.add_argument("--blocked", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--completed", type=str, default="{}")
    parser.add_argument("--extra-keys", type=str, default="")
    args = parser.parse_args()
    
    sampler = ModelSampler(extra_keys_str=args.extra_keys)
    sampler.run(args.samples, args.api_weights, args.model_weights, args.blocked, args.exclude, args.completed)