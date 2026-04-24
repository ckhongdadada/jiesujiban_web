from __future__ import annotations

import os
import time
import threading
from typing import Any, Generator

import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SpeculativeDecoder:
    """推测解码器 - 使用小模型加速大模型生成"""
    
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str | None = None,
        device: str = "cuda",
        draft_tokens: int = 5,
        temperature: float = 1.0
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers库未安装")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.draft_tokens = draft_tokens
        self.temperature = temperature
        
        self.target_model = None
        self.target_tokenizer = None
        self.draft_model = None
        self.draft_tokenizer = None
        
        self._target_model_path = target_model_path
        self._draft_model_path = draft_model_path
        
        self._loaded = False
        self._lock = threading.Lock()
        
        self.stats = {
            'total_generations': 0,
            'total_tokens_generated': 0,
            'total_draft_tokens': 0,
            'total_accepted_tokens': 0,
            'total_time': 0.0
        }
    
    def load_models(self) -> bool:
        """加载模型"""
        with self._lock:
            if self._loaded:
                return True
            
            try:
                print(f"[推测解码] 加载目标模型: {self._target_model_path}")
                
                self.target_tokenizer = AutoTokenizer.from_pretrained(
                    self._target_model_path,
                    trust_remote_code=True
                )
                
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self._target_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.target_model.eval()
                
                if self._draft_model_path and os.path.exists(self._draft_model_path):
                    print(f"[推测解码] 加载草稿模型: {self._draft_model_path}")
                    
                    self.draft_tokenizer = AutoTokenizer.from_pretrained(
                        self._draft_model_path,
                        trust_remote_code=True
                    )
                    
                    self.draft_model = AutoModelForCausalLM.from_pretrained(
                        self._draft_model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self.draft_model.eval()
                    
                    print("[推测解码] 草稿模型加载完成，启用推测解码")
                else:
                    print("[推测解码] 未找到草稿模型，将使用标准生成")
                    self.draft_model = None
                
                self._loaded = True
                return True
            
            except Exception as e:
                print(f"[推测解码] 模型加载失败: {e}")
                return False
    
    def _sample_from_distribution(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """从分布中采样"""
        if temperature == 0:
            return torch.argmax(logits, dim=-1).item()
        
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> str:
        """生成文本"""
        if not self._loaded:
            if not self.load_models():
                raise RuntimeError("模型加载失败")
        
        start_time = time.time()
        
        input_ids = self.target_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        generated_tokens = 0
        draft_generated = 0
        accepted_tokens = 0
        
        with torch.no_grad():
            while generated_tokens < max_new_tokens:
                if self.draft_model is not None:
                    draft_outputs = self.draft_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=min(self.draft_tokens, max_new_tokens - generated_tokens),
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        pad_token_id=self.target_tokenizer.pad_token_id,
                        eos_token_id=self.target_tokenizer.eos_token_id
                    )
                    
                    draft_new_tokens = draft_outputs[0, input_ids.shape[1]:]
                    draft_generated += len(draft_new_tokens)
                    
                    if len(draft_new_tokens) == 0:
                        break
                    
                    verify_input = torch.cat([input_ids, draft_new_tokens.unsqueeze(0)], dim=1)
                    verify_attention = torch.ones(1, verify_input.shape[1], device=self.device)
                    
                    target_outputs = self.target_model(
                        verify_input,
                        attention_mask=verify_attention,
                        use_cache=True
                    )
                    target_logits = target_outputs.logits
                    
                    accepted = []
                    
                    for i, draft_token in enumerate(draft_new_tokens):
                        pos = input_ids.shape[1] + i - 1
                        
                        if pos < target_logits.shape[1]:
                            if do_sample:
                                target_token = self._sample_from_distribution(
                                    target_logits[0, pos],
                                    temperature
                                )
                            else:
                                target_token = torch.argmax(target_logits[0, pos]).item()
                            
                            if target_token == draft_token.item():
                                accepted.append(draft_token.item())
                                accepted_tokens += 1
                            else:
                                accepted.append(target_token)
                                break
                        else:
                            accepted.append(draft_token.item())
                    
                    if accepted:
                        accepted_tensor = torch.tensor([accepted], device=self.device)
                        input_ids = torch.cat([input_ids, accepted_tensor.unsqueeze(0)], dim=1)
                        attention_mask = torch.ones(1, input_ids.shape[1], device=self.device)
                        generated_tokens += len(accepted)
                    
                    if generated_tokens >= max_new_tokens:
                        break
                
                else:
                    outputs = self.target_model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=min(10, max_new_tokens - generated_tokens),
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        pad_token_id=self.target_tokenizer.pad_token_id,
                        eos_token_id=self.target_tokenizer.eos_token_id
                    )
                    
                    new_tokens = outputs[0, input_ids.shape[1]:]
                    if len(new_tokens) == 0:
                        break
                    
                    input_ids = outputs
                    attention_mask = torch.ones(1, input_ids.shape[1], device=self.device)
                    generated_tokens += len(new_tokens)
        
        result = self.target_tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        elapsed = time.time() - start_time
        
        self.stats['total_generations'] += 1
        self.stats['total_tokens_generated'] += generated_tokens
        self.stats['total_draft_tokens'] += draft_generated
        self.stats['total_accepted_tokens'] += accepted_tokens
        self.stats['total_time'] += elapsed
        
        return result
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0
    ) -> Generator[str, None, None]:
        """流式生成文本"""
        if not self._loaded:
            if not self.load_models():
                raise RuntimeError("模型加载失败")
        
        from transformers import TextIteratorStreamer
        
        streamer = TextIteratorStreamer(
            self.target_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        input_ids = self.target_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        import threading
        
        thread = threading.Thread(
            target=self.target_model.generate,
            kwargs={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_new_tokens': max_new_tokens,
                'do_sample': do_sample,
                'temperature': temperature if do_sample else 1.0,
                'streamer': streamer,
                'pad_token_id': self.target_tokenizer.pad_token_id,
                'eos_token_id': self.target_tokenizer.eos_token_id
            }
        )
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats['total_generations'] > 0:
            stats['avg_tokens_per_generation'] = (
                stats['total_tokens_generated'] / stats['total_generations']
            )
            stats['avg_time_per_generation'] = (
                stats['total_time'] / stats['total_generations']
            )
            stats['tokens_per_second'] = (
                stats['total_tokens_generated'] / stats['total_time']
                if stats['total_time'] > 0 else 0
            )
        
        if stats['total_draft_tokens'] > 0:
            stats['acceptance_rate'] = (
                stats['total_accepted_tokens'] / stats['total_draft_tokens']
            )
        else:
            stats['acceptance_rate'] = 0
        
        stats['speculative_decoding_enabled'] = self.draft_model is not None
        
        return stats


_decoder_instance: SpeculativeDecoder | None = None
_decoder_lock = threading.Lock()


def get_speculative_decoder(
    target_model_path: str | None = None,
    draft_model_path: str | None = None,
    device: str = "cuda"
) -> SpeculativeDecoder:
    """获取推测解码器单例"""
    global _decoder_instance
    
    with _decoder_lock:
        if _decoder_instance is None:
            if target_model_path is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                target_model_path = os.getenv(
                    "GENERATOR_MODEL_PATH",
                    os.path.join(base_dir, "models", "generator")
                )
            
            _decoder_instance = SpeculativeDecoder(
                target_model_path=target_model_path,
                draft_model_path=draft_model_path,
                device=device
            )
        
        return _decoder_instance


def generate_with_speculative_decoding(
    prompt: str,
    max_new_tokens: int = 100,
    do_sample: bool = False,
    temperature: float = 1.0,
    target_model_path: str | None = None,
    draft_model_path: str | None = None
) -> str:
    """使用推测解码生成文本"""
    decoder = get_speculative_decoder(target_model_path, draft_model_path)
    return decoder.generate(prompt, max_new_tokens, do_sample, temperature)
