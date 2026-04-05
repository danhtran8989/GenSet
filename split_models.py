import argparse
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional

class ModelSampler:
    def __init__(self, base_env: str = ".env", extra_files: List[str] = None, extra_keys_str: str = ""):
        self.all_keys: Dict[str, str] = {}
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
                    self.all_keys[k.strip()] = v.strip()

    def _parse_extra_keys_str(self, extra_keys_str: str):
        if not extra_keys_str: return
        pairs = extra_keys_str.split(',')
        for pair in pairs:
            if '=' in pair:
                k, v = pair.split('=', 1)
                self.all_keys[k.strip()] = v.strip()

    @staticmethod
    def _parse_dict_arg(input_str: Optional[str]) -> Any:
        if not input_str or input_str == "{}": return {}
        try: return ast.literal_eval(input_str)
        except: return {}

    def run(self, num_samples: int, api_weights_str: str, model_weights_str: str, blocked_str: str, exclude_str: str, mode: str):
        model_cfg = self._parse_dict_arg(model_weights_str)
        specified_api_weights = self._parse_dict_arg(api_weights_str)
        
        blocked_keys = set([k.strip() for k in blocked_str.split(',') if k.strip()])
        exclude_keys = set([k.strip() for k in exclude_str.split(',') if k.strip()])
        
        active_platforms = [p.lower() for p in model_cfg.keys()]
        
        # 1. Deduplicate Keys by Value
        unique_platform_keys = {p: [] for p in active_platforms}
        for p in active_platforms:
            raw_list = [(k, v) for k, v in self.all_keys.items() if p in k.lower()]
            seen_values = {}
            for name, val in raw_list:
                if val not in seen_values:
                    seen_values[val] = name
                    unique_platform_keys[p].append(name)
                else:
                    # Sync weights if duplicate name was used in args
                    survivor = seen_values[val]
                    if name in specified_api_weights: specified_api_weights[survivor] = specified_api_weights[name]

        all_rel_keys = [k for keys in unique_platform_keys.values() for k in keys]
        if not all_rel_keys: return

        # 2. Initial Weight Assignment
        sum_spec = sum(specified_api_weights.get(k, 0.0) for k in all_rel_keys)
        unspec_keys = [k for k in all_rel_keys if k not in specified_api_weights]
        base_share = (max(0.0, 1.0 - sum_spec) / len(unspec_keys)) if unspec_keys else 0.0
        
        initial_weights = {k: specified_api_weights.get(k, base_share) for k in all_rel_keys}

        # 3. Mode Processing
        final_api_weights = {}
        redist_bonus = 0.0

        if mode == "redistribute":
            # Keys that will receive extra load (Not blocked AND not excluded)
            redist_targets = [k for k in all_rel_keys if k not in blocked_keys and k not in exclude_keys]
            # Weight harvested from blocked keys
            blocked_weight = sum(initial_weights[k] for k in all_rel_keys if k in blocked_keys)
            print("redist_targets", redist_targets)
            print("blocked_weight", blocked_weight)

            if redist_targets:
                redist_bonus = blocked_weight / len(redist_targets)
            
            for k in all_rel_keys:
                if k in blocked_keys:
                    continue # KEY IS GONE
                elif k in exclude_keys:
                    final_api_weights[k] = initial_weights[k] # KEEPS ORIGINAL ONLY
                else:
                    final_api_weights[k] = initial_weights[k] + redist_bonus # ORIGINAL + BONUS
        else:
            # Subtract mode: simply remove blocked and excluded keys
            # (Note: In your request, KEY_1 was excluded but you wanted it kept, 
            # so subtract mode here follows the standard 'remove blocked' logic)
            final_api_weights = {k: initial_weights[k] for k in all_rel_keys if k not in blocked_keys}

        # 4. Model Normalization
        norm_models = {}
        for platform, models in model_cfg.items():
            m_sum = sum(w for w in models.values() if w is not None)
            m_unspec = [m for m, w in models.items() if w is None]
            m_rem = max(0.0, 1.0 - m_sum)
            norm_models[platform.lower()] = {
                m: (w if w is not None else (m_rem / len(m_unspec) if m_unspec else 0))
                for m, w in models.items()
            }

        # 5. Slots & Distribution
        slots = []
        for api_key, api_w in final_api_weights.items():
            platform = next((p for p in active_platforms if p in api_key.lower()), None)
            for m_name, m_w in norm_models[platform].items():
                raw = api_w * m_w * num_samples
                slots.append({"key": api_key, "model": m_name, "count": int(raw), "rem": raw - int(raw)})

        target = int(round(num_samples if mode == "redistribute" else sum(final_api_weights.values()) * num_samples))
        diff = target - sum(s['count'] for s in slots)
        if diff > 0 and slots:
            slots.sort(key=lambda x: x['rem'], reverse=True)
            for i in range(diff): slots[i % len(slots)]['count'] += 1

        # 6. Report
        print(f"\n--- Distribution (Mode: {mode.upper()}) ---")
        for k in sorted(final_api_weights.keys()):
            bonus_str = ""
            if mode == "redistribute":
                bonus = redist_bonus if k not in exclude_keys else 0.0
                bonus_str = f"({initial_weights[k]:.2%} + {bonus:.2%} = {final_api_weights[k]:.2%})"
            else:
                bonus_str = f"({final_api_weights[k]:.2%})"
            
            print(f"\n🔑 {k} {bonus_str}")
            for s in [s for s in slots if s['key'] == k]:
                print(f"   - {s['model']:20}: {s['count']:>6} samples")

        print(f"\n{'='*40}\nREAL TOTAL ASSIGNED: {sum(s['count'] for s in slots)}\n{'='*40}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--model-weights", type=str, required=True)
    parser.add_argument("--api-weights", type=str, default="{}")
    parser.add_argument("--blocked", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--mode", choices=["redistribute", "subtract"], default="redistribute")
    parser.add_argument("--env-files", type=str, default="")
    parser.add_argument("--extra-keys", type=str, default="")
    args = parser.parse_args()
    
    sampler = ModelSampler(extra_files=[f.strip() for f in args.env_files.split(',') if f.strip()], extra_keys_str=args.extra_keys)
    sampler.run(args.samples, args.api_weights, args.model_weights, args.blocked, args.exclude, args.mode)