import argparse
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional

class ModelSampler:
    def __init__(self, base_env: str = ".env", extra_files: List[str] = None, extra_keys_str: str = ""):
        self.all_env_keys: Dict[str, str] = {}
        self._load_from_file(base_env)
        if extra_files:
            for file_path in extra_files:
                self._load_from_file(file_path)
        self._parse_extra_keys_str(extra_keys_str)

    def _load_from_file(self, file_path: str):
        path = Path(file_path)
        if not path.exists(): return
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    self.all_env_keys[k.strip()] = v.strip()

    def _parse_extra_keys_str(self, extra_keys_str: str):
        if not extra_keys_str: return
        for pair in extra_keys_str.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                self.all_env_keys[k.strip()] = v.strip()

    @staticmethod
    def _parse_dict_arg(input_str: Optional[str]) -> Any:
        if not input_str or input_str == "{}": return {}
        try: return ast.literal_eval(input_str)
        except: return {}

    def run(self, num_samples: int, api_weights_str: str, model_weights_str: str, blocked_str: str, exclude_str: str, mode: str):
        model_cfg = self._parse_dict_arg(model_weights_str)
        specified_weights = self._parse_dict_arg(api_weights_str)
        blocked_keys = set([k.strip() for k in blocked_str.split(',') if k.strip()])
        exclude_keys = set([k.strip() for k in exclude_str.split(',') if k.strip()])
        
        providers_list = []
        all_unique_names = []

        # 1. Structure Keys: Group by Provider and Deduplicate by Value
        for provider in model_cfg.keys():
            provider_node = {"provider": provider.lower(), "api_keys": []}
            seen_values = {}
            
            # Find keys belonging to this provider
            for k, v in self.all_env_keys.items():
                if provider.lower() in k.lower():
                    if v not in seen_values:
                        # Only add if NOT blocked
                        if k not in blocked_keys:
                            seen_values[v] = k
                            provider_node["api_keys"].append(k)
                            all_unique_names.append(k)
                        else:
                            # We keep track of unique blocked keys for weight math
                            if v not in seen_values:
                                seen_values[v] = k
                                all_unique_names.append(k)
            
            providers_list.append(provider_node)

        if not all_unique_names:
            print("❌ No keys found."); return

        # 2. Initial Weight Calculation (Fair share across ALL unique keys)
        sum_spec = sum(specified_weights.get(k, 0.0) for k in all_unique_names)
        unspec_keys = [k for k in all_unique_names if k not in specified_weights]
        base_share = (max(0.0, 1.0 - sum_spec) / len(unspec_keys)) if unspec_keys else 0.0
        initial_weights = {k: specified_weights.get(k, base_share) for k in all_unique_names}

        # 3. Redistribution Math
        final_api_weights = {}
        if mode == "redistribute":
            # Harvest weight from blocked keys
            blocked_weight_total = sum(initial_weights[k] for k in all_unique_names if k in blocked_keys)
            # Targets = Not Blocked and Not Excluded
            targets = [k for k in all_unique_names if k not in blocked_keys and k not in exclude_keys]
            bonus = (blocked_weight_total / len(targets)) if targets else 0.0

            for k in all_unique_names:
                if k in blocked_keys: continue
                final_api_weights[k] = initial_weights[k] + (bonus if k not in exclude_keys else 0.0)
        else:
            final_api_weights = {k: initial_weights[k] for k in all_unique_names if k not in blocked_keys}

        # 4. Model Weights Normalization
        norm_models = {}
        for p, models in model_cfg.items():
            m_sum = sum(w for w in models.values() if w is not None)
            m_unspec = [m for m, w in models.items() if w is None]
            m_rem = max(0.0, 1.0 - m_sum)
            norm_models[p.lower()] = {m: (w if w is not None else (m_rem/len(m_unspec) if m_unspec else 0)) for m, w in models.items()}

        # 5. Distribute Samples
        slots = []
        for provider_node in providers_list:
            p_name = provider_node["provider"]
            for k in provider_node["api_keys"]:
                api_w = final_api_weights[k]
                for m_name, m_w in norm_models[p_name].items():
                    raw = api_w * m_w * num_samples
                    slots.append({"key": k, "model": m_name, "count": int(raw), "rem": raw - int(raw), "provider": p_name})

        # Rounding
        target = int(round(num_samples if mode == "redistribute" else sum(final_api_weights.values()) * num_samples))
        diff = target - sum(s['count'] for s in slots)
        if diff > 0 and slots:
            slots.sort(key=lambda x: x['rem'], reverse=True)
            for i in range(diff): slots[i % len(slots)]['count'] += 1

        # 6. Final Report Grouped by Provider Dictionary
        print(f"\n--- Distribution Report (Mode: {mode.upper()}) ---")
        for p_node in providers_list:
            p_name = p_node["provider"]
            print(f"\n📁 Provider: {p_name.upper()}")
            if not p_node["api_keys"]:
                print("   (No active keys)")
                continue

            for k in sorted(p_node["api_keys"]):
                share = final_api_weights[k]
                bonus_info = ""
                if mode == "redistribute" and k not in exclude_keys:
                    bonus_info = f" (+{bonus:.2%} bonus)"
                
                print(f"  🔑 {k}: {share:.2%}{bonus_info}")
                for s in [s for s in slots if s['key'] == k]:
                    if s['count'] > 0:
                        print(f"     - {s['model']:20}: {s['count']:>6} samples")

        print(f"\n{'='*45}\nTOTAL ASSIGNED: {sum(s['count'] for s in slots)}\n{'='*45}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--model-weights", type=str, required=True)
    parser.add_argument("--api-weights", type=str, default="{}")
    parser.add_argument("--blocked", type=str, default="", help="Keys to remove and redistribute weight from")
    parser.add_argument("--exclude", type=str, default="", help="Keys to keep but NOT give bonus weight to")
    parser.add_argument("--mode", choices=["redistribute", "subtract"], default="redistribute")
    parser.add_argument("--env-files", type=str, default="")
    parser.add_argument("--extra-keys", type=str, default="")
    args = parser.parse_args()
    
    sampler = ModelSampler(extra_files=[f.strip() for f in args.env_files.split(',') if f.strip()], extra_keys_str=args.extra_keys)
    sampler.run(args.samples, args.api_weights, args.model_weights, args.blocked, args.exclude, args.mode)