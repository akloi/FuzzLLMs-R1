from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os

class LLMGenerator:
    def __init__(self, 
                model_name: str = "",
                temperature: float = 1,
                max_new_tokens: int = 512,
                batch_size: int = 8,
                gpu_devices: List[int] = None):
        if gpu_devices is None:
            gpu_devices = [0]
        
        self.gpu_devices = gpu_devices
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set CUDA visible devices to handle both single and multi-GPU scenarios uniformly
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_devices))
        self.device = "cuda:0"  # In visible devices, the first device is always 0
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically distribute to visible GPUs
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # 性能优化选项
            attn_implementation="flash_attention_2" if self._has_flash_attention() else None,
        ).eval()
        
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens  # 改用 max_new_tokens
        self.batch_size = batch_size
    
    def _has_flash_attention(self) -> bool:
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def extract_code(self, completion):
        """Extract code from completion"""
        match = re.search(r"<code>(.+?)</code>", completion, flags=re.DOTALL)
        if match:
            code = match.group(1).strip()
            code = code.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            return code
        return None
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """
        Args:
            prompts: 提示列表
            
        Returns:
            生成的代码列表
        """
        if not prompts:
            return []
        
        results = []

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_results = self._generate_batch_internal(batch_prompts)
            results.extend(batch_results)
        
        return results
    
    def _generate_batch_internal(self, prompts: List[str]) -> List[str]:
        """
        内部批处理生成函数
        """
        try:
            # 批量 tokenize
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=2048  # 输入最大长度
            ).to(self.device)
            
            # 记录每个 prompt 的原始长度（用于后续截取）
            prompt_lengths = [len(self.tokenizer.encode(p, add_special_tokens=True)) for p in prompts]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=self.max_new_tokens,  # 使用 max_new_tokens
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # 性能优化：减少不必要的计算
                    use_cache=True,
                )
            
            # 批量解码
            results = []
            for i, output in enumerate(outputs):
                # 只解码新生成的部分
                generated_ids = output[prompt_lengths[i]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                code = self.extract_code(response)
                results.append(code)
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return results
            
        except torch.cuda.OutOfMemoryError:
            print(f"[WARNING] GPU OOM with batch_size={len(prompts)}, falling back to smaller batches")
            
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 如果批次大小为1还OOM，返回None
            if len(prompts) == 1:
                return [None]
            
            # 递归地分成更小的批次
            mid = len(prompts) // 2
            left_results = self._generate_batch_internal(prompts[:mid])
            right_results = self._generate_batch_internal(prompts[mid:])
            return left_results + right_results
        
        except Exception as e:
            print(f"[ERROR] Error during batch generation: {e}")
            import traceback
            traceback.print_exc()
            # 返回与输入相同数量的None
            return [None] * len(prompts)

