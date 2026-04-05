import gradio as gr
import ast
import time
import io
import sys
from typing import Dict, List, Any, Optional

class ModelSamplerGradio:
    def __init__(self, extra_keys_str: str = ""):
        self.all_env_keys: Dict[str, str] = {}
        if extra_keys_str:
            for pair in extra_keys_str.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    self.all_env_keys[k.strip()] = v.strip()

    def run_sim(self, num_samples: int, api_weights_str: str, model_weights_str: str, 
                blocked_str: str, exclude_str: str, completed_str: str, progress=gr.Progress()):
        
        output_buffer = io.StringIO()
        
        try:
            model_cfg = ast.literal_eval(model_weights_str)
            spec_weights = ast.literal_eval(api_weights_str)
            partial_done = ast.literal_eval(completed_str)
        except Exception as e:
            return f"Error parsing inputs: {str(e)}"

        blocked_keys = set([k.strip() for k in blocked_str.split(',') if k.strip()])
        exclude_keys = set([k.strip() for k in exclude_str.split(',') if k.strip()])
        
        # 1. Key Discovery
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
        norm_models = {}
        for p, models in model_cfg.items():
            p_low = p.lower()
            norm_models[p_low] = {}
            fixed_sum = sum(v for v in models.values() if v is not None)
            none_count = len([mk for mk, mv in models.items() if mv is None])
            none_share = max(0.0, 1.0 - fixed_sum) / none_count if none_count > 0 else 0
            for m, w in models.items():
                norm_models[p_low][m] = w if w is not None else none_share

        # 5. Build Initial Queue
        queue = []
        key_totals = {} 
        for p_name, keys in unique_keys_by_provider.items():
            for k in keys:
                if k not in final_weights: continue
                for m_name, m_w in norm_models[p_name.lower()].items():
                    count = int(round(final_weights[k] * m_w * num_samples))
                    queue.append({"key": k, "model": m_name, "target": count, "provider": p_name})
                    key_totals[k] = key_totals.get(k, 0) + count

        output_buffer.write(f"🚀 Starting Simulation: {num_samples} samples\n")
        output_buffer.write(f"Active Keys: {', '.join(key_totals.keys())}\n\n")
        
        # 6. Execute Simulation
        results = {}
        total_processed = 0
        
        while queue:
            task = queue.pop(0)
            k, m, target = task['key'], task['model'], task['target']
            limit = partial_done.get(k, {}).get(m, float('inf'))
            
            processed = 0
            for _ in range(target):
                if processed >= limit:
                    rem = target - processed
                    failover_targets = [tk for tk in unique_keys_by_provider[task['provider']] 
                                        if tk not in blocked_keys and tk not in exclude_keys and tk != k]
                    
                    if failover_targets:
                        target_key = failover_targets[0]
                        queue.append({"key": target_key, "model": m, "target": rem, "provider": task['provider']})
                        output_buffer.write(f"⚠️ Failover: {k} reached limit for {m}. Moving {rem} samples to {target_key}\n")
                    break
                
                # Faster sleep for simulation
                time.sleep(0.0001) 
                processed += 1
                total_processed += 1
                
                if total_processed % 50 == 0:
                    progress(total_processed / num_samples, desc=f"Processing {k} | {m}")
            
            results.setdefault(k, {}).setdefault(m, 0)
            results[k][m] += processed
        
        # 7. Final Report
        output_buffer.write("\n" + "="*30 + "\n")
        output_buffer.write("--- FINAL DISTRIBUTION ---")
        output_buffer.write("\n" + "="*30 + "\n")
        
        for p_name in model_cfg.keys():
            output_buffer.write(f"\n📁 Provider: {p_name.upper()}\n")
            for k in sorted(all_unique_names):
                if p_name.lower() in k.lower() and k in results:
                    status = "❌" if k in blocked_keys else "🔑"
                    output_buffer.write(f"  {status} {k}:\n")
                    for m, count in results[k].items():
                        output_buffer.write(f"     - {m:20}: {count:>6} samples\n")

        return output_buffer.getvalue()

def launch_ui():
    with gr.Blocks(title="Model Sampler Simulator") as demo:
        gr.Markdown("# 🤖 Model Sampler & Failover Simulator")
        gr.Markdown("Test how keys are distributed and how failover works when an API key hits a limit.")
        
        with gr.Row():
            with gr.Column():
                samples = gr.Number(label="Number of Samples", value=5000)
                extra_keys = gr.Textbox(
                    label="Environment Keys (CSV key=val)", 
                    value="OLLAMA_1=v1, OLLAMA_2=v2, OPENAI_1=v3, OPENAI_2=v4",
                    placeholder="KEY_1=val, KEY_2=val"
                )
                model_weights = gr.Textbox(
                    label="Model Weights (Python Dict)", 
                    lines=4,
                    value="{\n  'ollama': {'llama3': 0.6, 'mistral': None},\n  'openai': {'gpt-4': 1.0}\n}"
                )
                api_weights = gr.Textbox(
                    label="Specific API Key Weights (Optional)", 
                    value="{'OLLAMA_1': 0.8}"
                )
                
            with gr.Column():
                blocked = gr.Textbox(label="Blocked Keys (CSV)", value="OPENAI_1", placeholder="OLLAMA_1, OPENAI_2")
                exclude = gr.Textbox(label="Exclude from Redistribution (CSV)", value="", placeholder="OLLAMA_2")
                completed = gr.Textbox(
                    label="Mid-Run Failures / Limits (Completed Counts)", 
                    lines=4,
                    value="{\n  'OLLAMA_1': {'llama3': 500}\n}"
                )
                btn = gr.Button("🚀 Run Simulation", variant="primary")

        output = gr.Textbox(label="Simulation Report", lines=20, interactive=False)

        def handle_run(n, api_w, mod_w, blk, exc, comp, keys):
            sampler = ModelSamplerGradio(extra_keys_str=keys)
            return sampler.run_sim(n, api_w, mod_w, blk, exc, comp)

        btn.click(
            fn=handle_run, 
            inputs=[samples, api_weights, model_weights, blocked, exclude, completed, extra_keys], 
            outputs=output
        )

    demo.launch()

if __name__ == "__main__":
    launch_ui()