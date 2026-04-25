"""
inference_benchmark.py
======================
Inference + quality benchmarking script for the fine-tuned 政务回复 model.

Metrics computed per sample and in aggregate:
  • ROUGE-1 / ROUGE-2 / ROUGE-L  (lexical overlap)
  • BLEU-4                        (n-gram precision)
  • 语义相似度 (Semantic Similarity) (deep-learning cosine similarity via sentence embeddings)
  • LLM-as-Judge scores           (Claude evaluating 5 dimensions, each 1–5)
  • Length statistics             (generated vs reference)
  • Keyword coverage rate        (domain-specific key terms)
  • Low-quality flag rate        (same filters used during training)

Semantic similarity uses a dedicated Chinese sentence-embedding model that runs on
CPU so it never conflicts with the 4-bit LLM occupying GPU memory.  Two backends
are tried in order:
  1. sentence-transformers library  (pip install sentence-transformers)
  2. Raw HuggingFace AutoModel with mean-pooling  (always available)

LLM-as-Judge uses the Anthropic API (Claude) to score each reply on:
  相关性     — Does the reply address the actual complaint?
  完整性     — Are the key facts and actions fully covered?
  专业性     — Is the tone appropriate for a government response?
  格式合规   — No greetings / signatures / boilerplate?
  可操作性   — Are concrete actions or outcomes stated?

Requires:  pip install anthropic
Enable via:  --use_judge  (add --api_key sk-ant-... or set ANTHROPIC_API_KEY env var)

Default embedding model: shibing624/text2vec-base-chinese
Override via --embed_model or EMBED_MODEL config below.

Results are written to:
  <OUTPUT_DIR>/benchmark_results.xlsx   — per-sample table (all metrics)
  <OUTPUT_DIR>/benchmark_summary.txt   — aggregate stats
  <OUTPUT_DIR>/judge_scores.xlsx        — detailed judge verdicts (if --use_judge)
"""

import os
import re
import json
import time
import argparse
import warnings
import pandas as pd
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  ⚙️  Configuration  (mirror your training cfg)
# ─────────────────────────────────────────────
BASE_MODEL_NAME = "C:/Users/28414/PycharmProjects/接诉即办/qwen_models/Qwen/Qwen2___5-1___5B-Instruct"
FINETUNED_DIR   = r"C:\Users\28414\Desktop\qwen_reply_model"   # your OUTPUT_DIR
DATA_FILE       = r"C:\Users\28414\Desktop\合并后数据 - 副本.xlsx"
OUTPUT_DIR      = r"C:\Users\28414\Desktop\qwen_reply_model\test1"                                  # results go here

# Sentence-embedding model for semantic similarity
# Can be a HuggingFace model ID or a local path.
# Recommended Chinese models (pick one):
#   "shibing624/text2vec-base-chinese"           ← default, strong Chinese STS
#   "BAAI/bge-small-zh-v1.5"                     ← lighter, slightly faster
#   "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  ← multilingual
EMBED_MODEL     = "shibing624/text2vec-base-chinese"
EMBED_BATCH_SIZE = 64     # how many texts to encode per GPU/CPU batch
EMBED_DEVICE    = "cpu"   # keep on CPU to avoid conflicting with the 4-bit LLM on GPU

# ── LLM-as-Judge (Anthropic / Claude) ────────────────────────
# Set your key here OR export ANTHROPIC_API_KEY in your shell.
ANTHROPIC_API_KEY  = ""                        # leave blank to use env var
JUDGE_MODEL        = "claude-sonnet-4-20250514" # model used as judge
JUDGE_SAMPLE_SIZE  = 50    # judge is slower/costly; evaluate a subset
JUDGE_MAX_RETRIES  = 3     # retries on transient API errors
JUDGE_RETRY_DELAY  = 5     # seconds between retries
JUDGE_RPM_PAUSE    = 1.0   # seconds between requests (rate-limit buffer)

# Inference knobs
SAMPLE_SIZE     = None    # how many rows to evaluate; set to None for full dataset
BATCH_SIZE      = 1      # keep at 1 for 4-bit quant
MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.1    # low temp → more deterministic for eval
TOP_P           = 0.9
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
#  Reuse exact same data-cleaning from training
# ─────────────────────────────────────────────
LOW_QUALITY_PATTERNS = [
    r'正在.*?研究', r'正在.*?推进', r'请.*?耐心等待',
    r'已转.*?部门', r'已转办', r'请.*?关注.*?进展',
]
ONSITE_NEGATIVE_PATTERNS = [
    r'经.*?核查.*?未发现', r'经.*?核实.*?未发现',
    r'经.*?现场.*?未.*?发现', r'现场.*?查看.*?未.*?发现',
    r'经.*?核查.*?无.*?问题', r'经.*?核实.*?符合.*?标准',
    r'未见.*?异常', r'无明显.*?问题',
]
REPLY_MIN_LEN = 50
REPLY_MAX_LEN = 600


def clean_training_data(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r'^[\s\S]*?(?:关于.*?回复信|关于.*?的函)\s*\n', '', text, flags=re.DOTALL)
    text = re.sub(r'您于[\d年月日]+.*?(?:现答复如下|回复如下)[：:。]?\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'您的留言.*?[，,。！!\n]\s*', '', text, flags=re.DOTALL)
    text = re.sub(r'^(?:尊敬的.*?[，,！!\n]|您好\s*[！!，,：:\n]|你好\s*[！!，,：:\n])\s*', '', text, flags=re.DOTALL)
    tail_patterns = [
        r'感谢您对.*?(?:理解[与和]?支持|关心[与和]?支持|关注[与和]?支持).*$',
        r'感谢您的.*?(?:理解|支持|关注|关心).*$',
        r'特此回复.*$', r'祝您.*?愉快.*$', r'请.*?谅解.*$',
        r'如有.*?疑问.*?联系.*$', r'欢迎.*?再次.*?留言.*$',
        r'\n\s*[\u4e00-\u9fa5]{2,20}(?:委员会|办公室|管理局|管理委|指挥中心|街道办|镇政府|局|处|科)\s*\n[\s\S]*$',
        r'\n\s*\d{4}年\d{1,2}月\d{1,2}日\s*$',
    ]
    for pat in tail_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'^[\s：:,，。！!\n]+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_low_quality(text: str) -> bool:
    for pat in LOW_QUALITY_PATTERNS:
        if re.search(pat, text) and len(text) < 120:
            return True
    return False


def is_onsite_negative(text: str) -> bool:
    return any(re.search(p, text) for p in ONSITE_NEGATIVE_PATTERNS)


# ─────────────────────────────────────────────
#  📊  Metric helpers
# ─────────────────────────────────────────────
def tokenize_zh(text: str) -> list:
    """Character-level tokenizer (no external dependency needed for Chinese)."""
    return list(text.replace(' ', '').replace('\n', ''))


def rouge_n(hyp_tokens: list, ref_tokens: list, n: int) -> float:
    """Recall-oriented ROUGE-N (standard definition)."""
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    ref_ng = ngrams(ref_tokens, n)
    hyp_ng = ngrams(hyp_tokens, n)
    if not ref_ng:
        return 0.0
    ref_counts: dict = {}
    for g in ref_ng:
        ref_counts[g] = ref_counts.get(g, 0) + 1
    match = sum(min(hyp_ng.count(g), cnt) for g, cnt in ref_counts.items())
    return match / len(ref_ng)


def rouge_l(hyp_tokens: list, ref_tokens: list) -> float:
    """ROUGE-L via LCS length."""
    m, n = len(hyp_tokens), len(ref_tokens)
    if m == 0 or n == 0:
        return 0.0
    # Use 1-D DP to save memory
    prev = [0] * (n + 1)
    lcs = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if hyp_tokens[i-1] == ref_tokens[j-1]:
                curr[j] = prev[j-1] + 1
                lcs = max(lcs, curr[j])
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    # F-measure variant used by most ROUGE implementations
    precision = lcs / m if m else 0
    recall    = lcs / n if n else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_4(hyp_tokens: list, ref_tokens: list) -> float:
    """Corpus-level BLEU-4 for a single sentence pair with brevity penalty."""
    if len(hyp_tokens) == 0:
        return 0.0
    log_bp = min(0.0, 1 - len(ref_tokens) / len(hyp_tokens))
    score = 0.0
    valid_n = 0
    for n in range(1, 5):
        def ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        ref_ng = ngrams(ref_tokens, n)
        hyp_ng = ngrams(hyp_tokens, n)
        if not hyp_ng:
            break
        ref_counts: dict = {}
        for g in ref_ng:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        match = sum(min(hyp_ng.count(g), ref_counts.get(g, 0)) for g in set(hyp_ng))
        p = match / len(hyp_ng) if hyp_ng else 0
        score += np.log(p + 1e-10)
        valid_n += 1
    if valid_n == 0:
        return 0.0
    return float(np.exp(log_bp + score / valid_n))


# Domain keyword lists for coverage evaluation
DOMAIN_KEYWORDS = [
    '核实', '核查', '整改', '处理', '已', '责令', '通知',
    '上报', '联系', '协调', '解决', '完成', '落实',
    '维修', '整治', '修缮', '清理', '督促', '建议',
]


def keyword_coverage(text: str) -> float:
    """Fraction of domain keywords present in the text."""
    hits = sum(1 for kw in DOMAIN_KEYWORDS if kw in text)
    return hits / len(DOMAIN_KEYWORDS)


def has_greeting(text: str) -> bool:
    return bool(re.search(r'^(?:您好|你好|尊敬)', text.strip()))


def has_closing(text: str) -> bool:
    return bool(re.search(r'感谢您|特此回复|祝您', text))


# ─────────────────────────────────────────────
#  🧠  Deep-learning semantic similarity
# ─────────────────────────────────────────────
class SemanticSimilarityScorer:
    """
    Computes cosine similarity between sentence embeddings.

    Backend priority:
      1. sentence-transformers  (cleanest API, auto mean-pool + normalise)
      2. HuggingFace AutoModel  (always available, manual mean-pool)

    Both backends run on EMBED_DEVICE (default: CPU) so they never
    compete with the 4-bit generation model for GPU memory.
    """

    def __init__(self, model_name: str = EMBED_MODEL, device: str = EMBED_DEVICE):
        self.model_name = model_name
        self.device     = device
        self._model     = None
        self._tokenizer = None
        self._backend   = None
        self._load()

    # ── loading ──────────────────────────────────────────────────────────
    def _load(self):
        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            print(f"\n🧠 Loading embedding model (sentence-transformers): {self.model_name}")
            self._model   = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sbert"
            print(f"   ✅ sentence-transformers backend ready on {self.device}")
            return
        except ImportError:
            print("   ℹ️  sentence-transformers not installed; falling back to AutoModel.")
        except Exception as e:
            print(f"   ⚠️  sentence-transformers failed ({e}); falling back to AutoModel.")

        # Fallback: raw HuggingFace AutoModel with mean-pooling
        try:
            from transformers import AutoTokenizer, AutoModel
            print(f"\n🧠 Loading embedding model (AutoModel): {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model     = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            self._backend   = "auto"
            print(f"   ✅ AutoModel backend ready on {self.device}")
        except Exception as e:
            print(f"   ❌ Failed to load embedding model: {e}")
            print("      Semantic similarity will be skipped (all values → NaN).")
            self._backend = "none"

    # ── encoding ─────────────────────────────────────────────────────────
    def _encode_sbert(self, texts: list[str]) -> np.ndarray:
        """Encode with sentence-transformers (returns L2-normalised embeddings)."""
        return self._model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _mean_pool(self, token_embeddings: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
        """Weighted mean pooling over non-padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, dim=1) / \
               torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    @torch.inference_mode()
    def _encode_auto(self, texts: list[str]) -> np.ndarray:
        """Encode with raw AutoModel + mean pooling + L2 normalisation."""
        all_vecs = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i: i + EMBED_BATCH_SIZE]
            enc   = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            out   = self._model(**enc)
            vecs  = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
            # L2 normalise
            vecs  = torch.nn.functional.normalize(vecs, p=2, dim=1)
            all_vecs.append(vecs.cpu().numpy())
        return np.vstack(all_vecs)

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """Public encode — returns (N, D) float32 array or None if unavailable."""
        if self._backend == "none":
            return None
        if self._backend == "sbert":
            return self._encode_sbert(texts)
        return self._encode_auto(texts)

    # ── similarity ───────────────────────────────────────────────────────
    def cosine_similarities(self,
                             hyp_texts: list[str],
                             ref_texts:  list[str]) -> list[float]:
        """
        Compute pairwise cosine similarity for two parallel lists of texts.
        Since embeddings are L2-normalised, cosine sim = dot product.

        Returns a list of floats in [-1, 1], or [NaN]*N if the model failed.
        """
        n = len(hyp_texts)
        if self._backend == "none":
            return [float("nan")] * n

        print(f"\n   🔢 Encoding {n} generated replies...")
        hyp_vecs = self.encode(hyp_texts)
        print(f"   🔢 Encoding {n} reference replies...")
        ref_vecs = self.encode(ref_texts)

        # Element-wise dot product (both already L2-normalised)
        sims = (hyp_vecs * ref_vecs).sum(axis=1).tolist()
        return [round(float(s), 4) for s in sims]


# ─────────────────────────────────────────────
#  ⚖️  LLM-as-Judge  (Claude via Anthropic API)
# ─────────────────────────────────────────────
# Scoring rubric — 5 dimensions, each rated 1–5
JUDGE_DIMENSIONS = {
    "相关性":   "回复是否直接回应了投诉/留言的核心诉求？（1=完全无关, 5=精准切题）",
    "完整性":   "是否涵盖了核实情况与办理结果，没有遗漏关键信息？（1=严重缺失, 5=完整全面）",
    "专业性":   "语言是否符合政务回复规范，措辞客观、准确、得体？（1=极不专业, 5=高度专业）",
    "格式合规": "是否避免了问候语、结束套话、署名日期等不必要内容？（1=格式混乱, 5=格式规范）",
    "可操作性": "是否明确说明了已采取或将要采取的具体措施？（1=空洞无措施, 5=措施具体明确）",
}

_JUDGE_SYSTEM = """\
你是一名严格、公正的政务回复质量评估专家。
你的任务是根据投诉内容和参考答案，对"生成回复"进行多维度打分。
请只输出 JSON，不要输出任何其他内容，不要添加 markdown 代码块标记。
"""

_JUDGE_USER_TMPL = """\
## 投诉信息
标签：{tag}
标题：{title}
正文：{body}

## 参考回复（人工撰写，供参考，不要照抄）
{reference}

## 待评估的生成回复
{generated}

## 评分任务
请对"生成回复"在以下5个维度各给出 1–5 的整数评分，并附一句简短理由（≤20字）。
维度说明：
{dimension_desc}

请严格按以下 JSON 格式输出，key 使用维度名称的中文，value 为包含 score 和 reason 的对象：
{{
  "相关性":   {{"score": <1-5>, "reason": "<≤20字>"}},
  "完整性":   {{"score": <1-5>, "reason": "<≤20字>"}},
  "专业性":   {{"score": <1-5>, "reason": "<≤20字>"}},
  "格式合规": {{"score": <1-5>, "reason": "<≤20字>"}},
  "可操作性": {{"score": <1-5>, "reason": "<≤20字>"}}
}}
"""


class LLMJudge:
    """
    Uses Claude (via the Anthropic API) to score generated 政务 replies
    on five domain-relevant dimensions, each rated 1–5.

    Usage:
        judge = LLMJudge(api_key="sk-ant-...")
        result = judge.score_one(tag, title, body, reference, generated)
        # result → {"相关性": {"score": 4, "reason": "..."}, ...}

        # Or score a whole DataFrame in batch:
        scores_df = judge.score_dataframe(df, gen_col="生成回复", ref_col="answer")
    """

    DIMENSIONS = list(JUDGE_DIMENSIONS.keys())
    DIM_DESC   = "\n".join(f"  {k}：{v}" for k, v in JUDGE_DIMENSIONS.items())

    def __init__(self,
                 api_key:   str  = "",
                 model:     str  = JUDGE_MODEL,
                 max_retries: int = JUDGE_MAX_RETRIES,
                 retry_delay: float = JUDGE_RETRY_DELAY,
                 rpm_pause:   float = JUDGE_RPM_PAUSE):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "pip install anthropic   ← required for LLM-as-Judge scoring"
            )
        key = api_key or ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "No Anthropic API key found.  Pass --api_key or set ANTHROPIC_API_KEY."
            )
        self._client      = _anthropic.Anthropic(api_key=key)
        self.model        = model
        self.max_retries  = max_retries
        self.retry_delay  = retry_delay
        self.rpm_pause    = rpm_pause

    # ── core scoring ────────────────────────────────────────────────────
    def _build_user_prompt(self, tag: str, title: str, body: str,
                           reference: str, generated: str) -> str:
        return _JUDGE_USER_TMPL.format(
            tag=tag, title=title, body=body,
            reference=reference, generated=generated,
            dimension_desc=self.DIM_DESC,
        )

    def _parse_response(self, raw: str) -> dict | None:
        """Extract and validate the JSON dict from Claude's response."""
        # Strip accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage by locating the first { ... }
            m = re.search(r'\{[\s\S]+\}', raw)
            if not m:
                return None
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return None

        # Validate structure
        for dim in self.DIMENSIONS:
            if dim not in data:
                return None
            entry = data[dim]
            if not isinstance(entry, dict) or "score" not in entry:
                return None
            # Clamp score to [1, 5]
            try:
                data[dim]["score"] = max(1, min(5, int(entry["score"])))
            except (TypeError, ValueError):
                return None
        return data

    def score_one(self, tag: str, title: str, body: str,
                  reference: str, generated: str) -> dict | None:
        """
        Score a single reply.  Returns a dict keyed by dimension name, or
        None if all retries fail.
        """
        user_msg = self._build_user_prompt(tag, title, body, reference, generated)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,    # fully deterministic judge
                    system=_JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": user_msg}],
                )
                raw = response.content[0].text
                parsed = self._parse_response(raw)
                if parsed is not None:
                    return parsed
                print(f"   ⚠️  Judge parse failed (attempt {attempt}), raw: {raw[:120]}")
            except Exception as e:
                print(f"   ⚠️  Judge API error (attempt {attempt}): {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        return None   # all retries exhausted

    def score_dataframe(self,
                        df: pd.DataFrame,
                        gen_col:  str = "生成回复",
                        ref_col:  str = "answer",
                        tag_col:  str = "留言标签",
                        title_col: str = "留言标题",
                        body_col:  str = "留言正文") -> pd.DataFrame:
        """
        Score every row in df; returns a new DataFrame aligned with df's index,
        containing one column per dimension score plus per-dimension reasons
        and an 综合得分 (mean score across all dimensions).
        """
        total = len(df)
        print(f"\n⚖️  LLM-as-Judge: scoring {total} samples with {self.model}...")
        print(f"   (Ctrl-C to stop early; results so far will still be saved)\n")

        score_records = []

        for i, (idx, row) in enumerate(df.iterrows(), 1):
            result = self.score_one(
                tag       = str(row.get(tag_col, "")),
                title     = str(row.get(title_col, "")),
                body      = str(row.get(body_col, "")),
                reference = str(row.get(ref_col, "")),
                generated = str(row.get(gen_col, "")),
            )

            rec = {"idx": idx}
            if result:
                for dim in self.DIMENSIONS:
                    rec[f"Judge_{dim}"]      = result[dim]["score"]
                    rec[f"Judge_{dim}_理由"] = result[dim].get("reason", "")
                scores = [result[dim]["score"] for dim in self.DIMENSIONS]
                rec["Judge_综合得分"] = round(sum(scores) / len(scores), 2)
            else:
                for dim in self.DIMENSIONS:
                    rec[f"Judge_{dim}"]      = None
                    rec[f"Judge_{dim}_理由"] = "scoring failed"
                rec["Judge_综合得分"] = None

            score_records.append(rec)

            # Progress
            valid = sum(1 for r in score_records if r["Judge_综合得分"] is not None)
            avg   = np.nanmean([r["Judge_综合得分"] for r in score_records
                                 if r["Judge_综合得分"] is not None]) if valid else float("nan")
            print(f"  [{i:>4}/{total}]  综合得分={rec['Judge_综合得分']}  "
                  f"running mean={avg:.2f}  ({valid} scored)")

            time.sleep(self.rpm_pause)   # polite rate-limit pause

        return pd.DataFrame(score_records).set_index("idx")
def load_eval_data(data_file: str, sample_size: int | None, seed: int) -> pd.DataFrame:
    print(f"\n📂 Loading data: {data_file}")
    df = pd.read_excel(data_file, engine='openpyxl') if data_file.endswith('.xlsx') else pd.read_csv(data_file)
    df.fillna('', inplace=True)

    df = df[df['官方回复正文'].str.len() > 5]
    df['answer'] = df['官方回复正文'].apply(clean_training_data)
    df = df[df['answer'].str.len().between(REPLY_MIN_LEN, REPLY_MAX_LEN)]
    df = df[~df['answer'].apply(is_low_quality)]
    df = df[~df['answer'].apply(is_onsite_negative)]
    df = df.drop_duplicates(subset=['answer'], keep='first').reset_index(drop=True)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    print(f"   Evaluation samples: {len(df):,}")
    return df


# ─────────────────────────────────────────────
#  🤖  Model loading
# ─────────────────────────────────────────────
def load_model_and_tokenizer(finetuned_dir: str, base_model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n🔧 Loading tokenizer from: {finetuned_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(finetuned_dir, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad for generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"🔧 Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"🔧 Loading LoRA adapter: {finetuned_dir}")
    model = PeftModel.from_pretrained(base_model, finetuned_dir)
    model.eval()
    print("✅ Model ready.\n")
    return model, tokenizer


# ─────────────────────────────────────────────
#  🏗️  Prompt builder  (mirrors training exactly)
# ─────────────────────────────────────────────
def build_prompt(row: pd.Series, tokenizer) -> str:
    user_content = f"标签: {row['留言标签']}\n标题: {row['留言标题']}\n正文: {row['留言正文']}"
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业的政务业务处理引擎。"
                "请直接输出核实情况和办理结果，"
                "不需要任何问候语、引导语和结束语，"
                "不需要署名或日期，控制在200字以内。"
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ─────────────────────────────────────────────
#  🔮  Single-sample inference
# ─────────────────────────────────────────────
@torch.inference_mode()
def generate_reply(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=900).to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=TEMPERATURE > 0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode only newly generated tokens
    new_ids = output_ids[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
#  📈  Main benchmark loop
# ─────────────────────────────────────────────
def run_benchmark(df: pd.DataFrame, model, tokenizer,
                  output_dir: str, embed_model: str = EMBED_MODEL,
                  judge: "LLMJudge | None" = None,
                  judge_sample_size: int = JUDGE_SAMPLE_SIZE):
    records   = []
    total     = len(df)
    gen_texts = []   # collect for batch semantic scoring
    ref_texts = []

    print(f"🏃 Running inference on {total} samples...\n")

    for idx, row in df.iterrows():
        prompt   = build_prompt(row, tokenizer)
        ref_text = row['answer']

        generated = generate_reply(prompt, model, tokenizer)

        # Tokenize for lexical metrics
        hyp_tok = tokenize_zh(generated)
        ref_tok = tokenize_zh(ref_text)

        r1  = rouge_n(hyp_tok, ref_tok, 1)
        r2  = rouge_n(hyp_tok, ref_tok, 2)
        rl  = rouge_l(hyp_tok, ref_tok)
        b4  = bleu_4(hyp_tok, ref_tok)
        kw  = keyword_coverage(generated)
        lq  = is_low_quality(generated)
        neg = is_onsite_negative(generated)
        greet  = has_greeting(generated)
        close  = has_closing(generated)
        gen_len = len(generated)
        ref_len = len(ref_text)

        gen_texts.append(generated)
        ref_texts.append(ref_text)

        records.append({
            "idx":          idx,
            "标签":          row.get('留言标签', ''),
            "标题":          row.get('留言标题', ''),
            "参考回复":      ref_text,
            "生成回复":      generated,
            "ROUGE-1":      round(r1, 4),
            "ROUGE-2":      round(r2, 4),
            "ROUGE-L":      round(rl, 4),
            "BLEU-4":       round(b4, 4),
            "语义相似度":    None,           # filled in after batch encode
            "关键词覆盖率":  round(kw, 4),
            "生成长度(字)":  gen_len,
            "参考长度(字)":  ref_len,
            "含问候语":      greet,
            "含结束套话":    close,
            "低质量标志":    lq,
            "现场否认标志":  neg,
        })

        # Progress log every 10 samples
        if len(records) % 10 == 0 or len(records) == total:
            avg_r1 = np.mean([r["ROUGE-1"] for r in records])
            avg_rl = np.mean([r["ROUGE-L"] for r in records])
            print(f"  [{len(records):>4}/{total}]  "
                  f"ROUGE-1={avg_r1:.3f}  ROUGE-L={avg_rl:.3f}  "
                  f"len={gen_len}")

    # ── Semantic similarity (single batch pass) ──────────────────────────
    print("\n📐 Computing deep-learning semantic similarity (batch)...")
    scorer = SemanticSimilarityScorer(model_name=embed_model)
    sim_scores = scorer.cosine_similarities(gen_texts, ref_texts)
    for rec, sim in zip(records, sim_scores):
        rec["语义相似度"] = sim
    avg_sem = np.nanmean(sim_scores)
    print(f"   ✅ Mean semantic similarity: {avg_sem:.4f}")

    # ── LLM-as-Judge (optional) ──────────────────────────────────────────
    results_df = pd.DataFrame(records)
    if judge is not None:
        # Sample a subset for judging (API cost control)
        judge_n   = min(judge_sample_size, len(results_df))
        judge_df  = results_df.sample(n=judge_n, random_state=RANDOM_SEED)

        # Attach the original complaint columns needed by the judge
        complaint_cols = ["留言标签", "留言标题", "留言正文"]
        available = [c for c in complaint_cols if c in df.columns]
        if available:
            judge_df = judge_df.join(df[available], on="idx", rsuffix="_orig")

        judge_scores = judge.score_dataframe(
            judge_df,
            gen_col   = "生成回复",
            ref_col   = "参考回复",
            tag_col   = "留言标签",
            title_col = "留言标题",
            body_col  = "留言正文",
        )

        # Save detailed judge verdicts
        judge_out = os.path.join(output_dir, "judge_scores.xlsx")
        full_judge = judge_df[["标签", "标题", "参考回复", "生成回复",
                                "ROUGE-L", "语义相似度"]].join(judge_scores)
        full_judge.to_excel(judge_out, index=False, engine="openpyxl")
        print(f"\n💾 Judge verdicts saved → {judge_out}")

        # Merge aggregate judge scores back into results_df for summary
        results_df = results_df.join(judge_scores, on="idx")
    else:
        # Fill judge columns with NaN so downstream summary is uniform
        for dim in LLMJudge.DIMENSIONS:
            results_df[f"Judge_{dim}"] = None
        results_df["Judge_综合得分"] = None
    os.makedirs(output_dir, exist_ok=True)
    out_xlsx = os.path.join(output_dir, "benchmark_results.xlsx")
    results_df.to_excel(out_xlsx, index=False, engine='openpyxl')
    print(f"\n💾 Per-sample results saved → {out_xlsx}")

    # ── Aggregate summary ──
    metric_cols = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-4", "关键词覆盖率"]
    summary_lines = [
        "=" * 60,
        "  📊  BENCHMARK SUMMARY",
        "=" * 60,
        f"  Samples evaluated  : {total}",
        f"  Embedding model    : {embed_model}",
        f"  Temperature        : {TEMPERATURE}",
        f"  Max new tokens     : {MAX_NEW_TOKENS}",
        "",
        "  ── Lexical Quality ──────────────────────────────────",
    ]
    for col in metric_cols:
        vals = results_df[col]
        summary_lines.append(
            f"  {col:<16}  mean={vals.mean():.4f}  "
            f"median={vals.median():.4f}  std={vals.std():.4f}"
        )

    # Semantic similarity block
    sem_vals = results_df["语义相似度"].dropna()
    sem_note = "(higher = more semantically aligned, range [-1, 1])"
    summary_lines += [
        "",
        "  ── Semantic Similarity (deep-learning cosine) ───────",
        f"  Model              : {embed_model}",
        f"  语义相似度         mean={sem_vals.mean():.4f}  "
        f"median={sem_vals.median():.4f}  std={sem_vals.std():.4f}",
        f"                     {sem_note}",
        f"  Samples scored     : {len(sem_vals)} / {total}",
        # Distribution buckets
        f"  ≥ 0.80 (excellent) : {(sem_vals >= 0.80).sum()} "
        f"({(sem_vals >= 0.80).mean():.1%})",
        f"  0.60–0.79 (good)   : {((sem_vals >= 0.60) & (sem_vals < 0.80)).sum()} "
        f"({((sem_vals >= 0.60) & (sem_vals < 0.80)).mean():.1%})",
        f"  0.40–0.59 (fair)   : {((sem_vals >= 0.40) & (sem_vals < 0.60)).sum()} "
        f"({((sem_vals >= 0.40) & (sem_vals < 0.60)).mean():.1%})",
        f"  < 0.40  (poor)     : {(sem_vals < 0.40).sum()} "
        f"({(sem_vals < 0.40).mean():.1%})",
    ]

    # LLM Judge summary block (only shown if judging was run)
    judge_综合 = results_df["Judge_综合得分"].dropna()
    if len(judge_综合) > 0:
        summary_lines += [
            "",
            "  ── LLM-as-Judge  (Claude, 1–5 each dimension) ──────",
            f"  Judge model        : {JUDGE_MODEL}",
            f"  Samples judged     : {len(judge_综合)} / {total}",
            f"  综合得分           mean={judge_综合.mean():.2f}  "
            f"median={judge_综合.median():.2f}  std={judge_综合.std():.2f}",
        ]
        for dim in LLMJudge.DIMENSIONS:
            col = f"Judge_{dim}"
            if col in results_df.columns:
                vals = results_df[col].dropna()
                if len(vals):
                    summary_lines.append(
                        f"  {dim:<8}           mean={vals.mean():.2f}  "
                        f"median={vals.median():.2f}  std={vals.std():.2f}"
                    )
        # Grade distribution for 综合得分
        summary_lines += [
            "",
            f"  ≥ 4.5 (优秀)       : {(judge_综合 >= 4.5).sum()} "
            f"({(judge_综合 >= 4.5).mean():.1%})",
            f"  3.5–4.4 (良好)     : {((judge_综合 >= 3.5) & (judge_综合 < 4.5)).sum()} "
            f"({((judge_综合 >= 3.5) & (judge_综合 < 4.5)).mean():.1%})",
            f"  2.5–3.4 (一般)     : {((judge_综合 >= 2.5) & (judge_综合 < 3.5)).sum()} "
            f"({((judge_综合 >= 2.5) & (judge_综合 < 3.5)).mean():.1%})",
            f"  < 2.5  (较差)      : {(judge_综合 < 2.5).sum()} "
            f"({(judge_综合 < 2.5).mean():.1%})",
        ]

    summary_lines += [
        "",
        "  ── Length Statistics ────────────────────────────────",
        f"  Generated (mean)   : {results_df['生成长度(字)'].mean():.1f} chars",
        f"  Reference (mean)   : {results_df['参考长度(字)'].mean():.1f} chars",
        f"  Generated (median) : {results_df['生成长度(字)'].median():.1f} chars",
        "",
        "  ── Format Compliance ────────────────────────────────",
        f"  含问候语 rate      : {results_df['含问候语'].mean():.1%}  (lower is better)",
        f"  含结束套话 rate    : {results_df['含结束套话'].mean():.1%}  (lower is better)",
        "",
        "  ── Quality Flags ────────────────────────────────────",
        f"  低质量 (low-qual)  : {results_df['低质量标志'].mean():.1%}",
        f"  现场否认           : {results_df['现场否认标志'].mean():.1%}",
        "=" * 60,
    ]

    # Top 5 best / worst by semantic similarity
    summary_lines += [
        "",
        "  ── Top-5 Best Samples (by 语义相似度) ──────────────",
    ]
    top5 = results_df.nlargest(5, "语义相似度")[
        ["标题", "语义相似度", "ROUGE-L", "BLEU-4", "生成长度(字)"]
    ]
    for _, r in top5.iterrows():
        summary_lines.append(
            f"  {str(r['标题'])[:28]:<30}  Sem={r['语义相似度']:.3f}"
            f"  RL={r['ROUGE-L']:.3f}  B4={r['BLEU-4']:.3f}"
        )

    summary_lines += [
        "",
        "  ── Top-5 Worst Samples (by 语义相似度) ─────────────",
    ]
    bot5 = results_df.nsmallest(5, "语义相似度")[
        ["标题", "语义相似度", "ROUGE-L", "BLEU-4", "生成长度(字)"]
    ]
    for _, r in bot5.iterrows():
        summary_lines.append(
            f"  {str(r['标题'])[:28]:<30}  Sem={r['语义相似度']:.3f}"
            f"  RL={r['ROUGE-L']:.3f}  B4={r['BLEU-4']:.3f}"
        )
    summary_lines.append("=" * 60)

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    out_txt = os.path.join(output_dir, "benchmark_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n💾 Summary saved → {out_txt}")

    return results_df


# ─────────────────────────────────────────────
#  🔬  Optional: compare base model vs fine-tuned
# ─────────────────────────────────────────────
def compare_with_base(df: pd.DataFrame, base_model_name: str, tokenizer,
                      output_dir: str, embed_model: str = EMBED_MODEL):
    """
    Run a small side-by-side comparison (first 20 samples) between the
    base model and the fine-tuned model. Useful for a quick sanity check.
    Call this after run_benchmark() if desired.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    print("\n🔬 Loading base model for comparison (first 20 samples)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, local_files_only=True,
    )
    base_model.eval()

    compare_rows = []
    base_gen_texts = []
    ref_texts_cmp  = []
    sample = df.head(20)

    for _, row in sample.iterrows():
        prompt   = build_prompt(row, tokenizer)
        ref      = row['answer']
        base_out = generate_reply(prompt, base_model, tokenizer)
        base_gen_texts.append(base_out)
        ref_texts_cmp.append(ref)
        compare_rows.append({
            "标题":     row.get('留言标题', ''),
            "参考回复": ref,
            "基座模型": base_out,
            "BASE_RL":  round(rouge_l(tokenize_zh(base_out), tokenize_zh(ref)), 4),
            "BASE_语义相似度": None,   # filled below
        })

    del base_model
    torch.cuda.empty_cache()

    # Semantic similarity for base model outputs
    print("📐 Computing semantic similarity for base model outputs...")
    scorer     = SemanticSimilarityScorer(model_name=embed_model)
    base_sims  = scorer.cosine_similarities(base_gen_texts, ref_texts_cmp)
    for row_dict, sim in zip(compare_rows, base_sims):
        row_dict["BASE_语义相似度"] = sim

    compare_df = pd.DataFrame(compare_rows)
    out = os.path.join(output_dir, "base_vs_finetuned_comparison.xlsx")
    compare_df.to_excel(out, index=False, engine='openpyxl')
    print(f"💾 Base vs fine-tuned comparison saved → {out}")
    print(f"   Base model avg ROUGE-L      (20 samples): {compare_df['BASE_RL'].mean():.4f}")
    print(f"   Base model avg 语义相似度   (20 samples): {np.nanmean(base_sims):.4f}")


# ─────────────────────────────────────────────
#  🚀  Entry point
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark the fine-tuned 政务 reply model")
    parser.add_argument("--finetuned_dir",   default=FINETUNED_DIR)
    parser.add_argument("--base_model",      default=BASE_MODEL_NAME)
    parser.add_argument("--data_file",       default=DATA_FILE)
    parser.add_argument("--output_dir",      default=OUTPUT_DIR)
    parser.add_argument("--sample_size",     type=int, default=SAMPLE_SIZE)
    parser.add_argument("--embed_model",     default=EMBED_MODEL,
                        help="HuggingFace model ID or local path for sentence embeddings")
    parser.add_argument("--use_judge",       action="store_true",
                        help="Enable LLM-as-Judge scoring via the Anthropic API (Claude)")
    parser.add_argument("--api_key",         default="",
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--judge_model",     default=JUDGE_MODEL,
                        help="Claude model to use as judge")
    parser.add_argument("--judge_n",         type=int, default=JUDGE_SAMPLE_SIZE,
                        help="Number of samples to judge (subset for cost control)")
    parser.add_argument("--compare_base",    action="store_true",
                        help="Also run inference with the base model for comparison")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.cuda.empty_cache()
    np.random.seed(RANDOM_SEED)

    df = load_eval_data(args.data_file, args.sample_size, RANDOM_SEED)
    model, tokenizer = load_model_and_tokenizer(args.finetuned_dir, args.base_model)

    # Optionally initialise the LLM judge
    judge = None
    if args.use_judge:
        print("\n⚖️  Initialising LLM-as-Judge...")
        judge = LLMJudge(
            api_key = args.api_key,
            model   = args.judge_model,
        )

    results_df = run_benchmark(df, model, tokenizer, args.output_dir,
                               embed_model=args.embed_model,
                               judge=judge,
                               judge_sample_size=args.judge_n)

    if args.compare_base:
        compare_with_base(df, args.base_model, tokenizer, args.output_dir,
                          embed_model=args.embed_model)

    print("\n✅ Benchmarking complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted by user.")
    except Exception:
        import traceback
        traceback.print_exc()