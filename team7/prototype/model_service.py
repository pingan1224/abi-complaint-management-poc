"""Run only on an allocated DGX GPU. Local-only classroom model service."""
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_ID = 'qwen2.5-32b-team7-base'
PATH = '/mnt/nas/models/approved/qwen2.5-32b-instruct-fp8'

def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, CompressedTensorsConfig
    if not torch.cuda.is_available(): raise RuntimeError('Allocated CUDA GPU required')
    torch.manual_seed(20261028)
    tokenizer = AutoTokenizer.from_pretrained(PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(PATH, local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 'cuda'}, low_cpu_mem_usage=True,
        quantization_config=CompressedTensorsConfig(dequantize=True))
    model.eval()
    print(json.dumps(dict(event='MODEL_READY', model=MODEL_ID, checkpoint=PATH,
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
        adapter=None, slurm_job=os.environ.get('SLURM_JOB_ID'))), flush=True)
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != '/v1/chat/completions': self.send_error(404); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 100000: raise ValueError('size')
                request = json.loads(self.rfile.read(size))
                if request['model'] != MODEL_ID: raise ValueError('model')
                prompt = tokenizer.apply_chat_template(request['messages'], tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=420, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                content = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                body = json.dumps(dict(model=MODEL_ID, choices=[dict(message=dict(content=content))])).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            except Exception:
                self.send_error(503, 'Model request unavailable')
    HTTPServer(('127.0.0.1', int(os.environ.get('TEAM7_PORT', '18707'))), Handler).serve_forever()

if __name__ == '__main__': main()
