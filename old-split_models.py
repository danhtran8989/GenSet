import argparse
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional
from xml.parsers.expat import model

class ModelSampler:
    def __init__(self, env_file: str = ".env"):
        self.env_file = env_file
        self.all_keys: Dict[str, str] = self._load_env_keys()

    def _load_env_keys(self) -> Dict[str, str]:
        env_path = Path(self.env_file)
        if not env_path.exists():
            return {
                "OLLAMA_API_KEY_1": "k1", "OLLAMA_API_KEY_2": "k2", "OLLAMA_API_KEY_3": "k3",
                "MISTRAL_API_KEY_1": "k4", "OPENAI_API_KEY_1": "k5", "OPENAI_API_KEY_2": "k6"
            }
        keys = {}
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    keys[k.strip()] = v.strip()
        return keys

    @staticmethod
    def _parse_dict_arg(input_str: Optional[str]) -> Any:
        if not input_str: 
            return {}
        try:
            return ast.literal_eval(input_str)
        except:
            return {}

    def run(self, num_samples: int, api_weights_str: str, model_weights_str: str, blocked_str: str, mode: str):
        model_cfg = self._parse_dict_arg(model_weights_str)
        specified_api_weights = self._parse_dict_arg(api_weights_str)
        blocked_keys = [k.strip() for k in blocked_str.split(',') if k.strip()] if blocked_str else []

        # 1. Identify relevant keys for the active platforms
        active_platforms = [p.lower() for p in model_cfg.keys()]
        
        # All keys that match the platform, regardless of status
        platform_keys = [
            k for k in self.all_keys.keys() 
            if any(p in k.lower() for p in active_platforms)
        ]

        if not platform_keys:
            print("Error: No API keys found for these platforms.")
            return

        # 2. Determine which keys participate in weight calculation
        # If 'redistribute', we only calculate weights for non-blocked keys
        # If 'subtract', we calculate weights for ALL, then drop blocked ones
        calculation_keys = [k for k in platform_keys if k not in blocked_keys] if mode == "redistribute" else platform_keys

        if not calculation_keys:
            print("Error: All keys are blocked and mode is set to 'redistribute'.")
            return

        # 3. API Weight Normalization
        api_weights_map = {k: specified_api_weights.get(k) for k in calculation_keys}
        sum_fixed = sum(w for w in api_weights_map.values() if w is not None)
        unspecified = [k for k, w in api_weights_map.items() if w is None]
        
        calculated_api_weights = {}
        if unspecified:
            remaining = max(0.0, 1.0 - sum_fixed)
            for k, w in api_weights_map.items():
                if w is not None:
                    calculated_api_weights[k] = w
                else:
                    calculated_api_weights[k] = remaining / len(unspecified)
        else:
            scale_factor = 1.0 / sum_fixed if sum_fixed > 0 else 1.0
            for k, w in api_weights_map.items():
                calculated_api_weights[k] = (w or 0) * scale_factor

        # 4. Filter for Final Distribution
        # If mode was 'subtract', we now drop the blocked keys from the final weights
        final_api_weights = {k: v for k, v in calculated_api_weights.items() if k not in blocked_keys}

        # 5. Model Weight Normalization
        norm_models = {}
        for platform, models in model_cfg.items():
            m_sum = sum(w for w in models.values() if w is not None)
            m_unspecified = [m for m, w in models.items() if w is None]
            m_rem = max(0.0, 1.0 - m_sum)
            norm_models[platform.lower()] = {
                m: (w if w is not None else (m_rem / len(m_unspecified) if m_unspecified else 0))
                for m, w in models.items()
            }

        # 6. Distribute Samples
        slots = []
        for api_key, api_w in final_api_weights.items():
            platform = next((p for p in norm_models if p in api_key.lower()), None)
            if not platform: continue
            for model, model_w in norm_models[platform].items():
                raw_val = api_w * model_w * num_samples
                slots.append({
                    "key": api_key, 
                    "model": model, 
                    "count": int(raw_val), 
                    "rem": raw_val - int(raw_val)
                })

        # 7. Safe Rounding Correction
        initial_total = sum(s['count'] for s in slots)
        
        # We only try to balance back to the original num_samples if we are in 'redistribute' mode
        # In 'subtract' mode, the total is naturally expected to be lower.
        target_total = num_samples if mode == "redistribute" else sum(final_api_weights.values()) * num_samples
        target_total = int(round(target_total))
        
        diff = target_total - initial_total
        
        if diff > 0 and slots:
            slots.sort(key=lambda x: x['rem'], reverse=True)
            for i in range(diff):
                slots[i % len(slots)]['count'] += 1

        real_total = sum(s['count'] for s in slots)

        # 8. Print Output
        print(f"\n--- Distribution Report (Mode: {mode.upper()}) ---")
        print(f"Requested: {num_samples} samples")
        if blocked_keys:
            print(f"Blocked Keys: {', '.join(blocked_keys)}")
        
        for k in sorted(final_api_weights.keys()):
            print(f"\n🔑 {k} (Weight: {final_api_weights[k]:.2%})")
            key_slots = [s for s in slots if s['key'] == k]
            for s in sorted(key_slots, key=lambda x: x['model']):
                if s['count'] > 0:
                    print(f"   - {s['model']:20}: {s['count']:>6} samples")

        print(f"\n{'='*65}")
        print(f"✅ REAL TOTAL SAMPLES ASSIGNED : {real_total}")
        print(f"   Target for this mode         : {target_total}")
        print(f"   Difference from Requested    : {real_total - num_samples}")
        print(f"{'='*65}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Model & API Key Sampler")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--model-weights", type=str, required=True,
                        help='Example: {"openai": {"gpt-4o": 0.6, "gpt-4o-mini": None}}')
    parser.add_argument("--api-weights", type=str, default="{}")
    parser.add_argument("--blocked", type=str, default="", help="Comma separated keys")
    parser.add_argument("--mode", choices=["redistribute", "subtract"], default="redistribute",
                        help="redistribute: share blocked key's load; subtract: lose blocked key's samples")
    
    args = parser.parse_args()
    ModelSampler().run(args.samples, args.api_weights, args.model_weights, args.blocked, args.mode)

    # # Result: 100 samples will go to OPENAI_API_KEY_2 (if available).
    # python split_models.py --samples 10000 \
    #     --model-weights "{'ollama': {'llama3': 0.5, 'gemma': 0.5}}" \
    #     --api-weights "{'OLLAMA_API_KEY_1': 0.3, 'OLLAMA_API_KEY_2': 0.6}" \
    #     --blocked "OLLAMA_API_KEY_3" --mode redistribute

    # # Result: If there are 2 keys, 50 samples are assigned to KEY_2, and the 50 from KEY_1 are discarded. Total = 50.
    # python split_models.py --samples 10000 \
    #     --model-weights "{'ollama': {'llama3': 0.5, 'gemma': 0.5}}" \
    #     --api-weights "{'OLLAMA_API_KEY_1': 0.3, 'OLLAMA_API_KEY_2': 0.6}" \
    #     --blocked "OLLAMA_API_KEY_3" --mode subtract
