# Đề xuất: Unified Fine-tuned Classifier + Calibrated Reject Rule

> Thay thế Stage 2 + Stage 3 của pipeline hiện tại (`shared/three_tier.py`, "Run 13 — mandatory escalation") bằng một classifier multi-class duy nhất, fine-tuned, với ngưỡng escalate có căn cứ thống kê thay cho luật bất đối xứng tay.

Tài liệu này đánh giá khách quan pipeline hiện đang chạy production và đề xuất hướng thiết kế lại Stage 2/3, độc lập với công sức đã đầu tư vào các vòng audit trước đó (Run 7 → Run 13).

---

## 1. Tóm tắt

Pipeline hiện tại (`Stage 2`: frozen BGE embedding + one-directional SVM gate; `Stage 3`: cosine agent-domain với threshold tay) hoạt động được, nhưng có hai vấn đề kỹ thuật ở gốc:

1. **Embedding frozen, không fine-tune theo domain** → SVM chỉ tự tin một chiều ("quality"), phải vá bằng luật bất đối xứng (`SVM_QUALITY_THRESH`, không bao giờ assert "quantity").
2. **Agent-domain context bị tách thành stage riêng với threshold tay** (`THRESH_IN_DOMAIN=0.55`) thay vì là một input feature của classifier.

Cả hai là **triệu chứng của một classifier yếu**, không phải đặc tính bắt buộc của bài toán. Đề xuất: gộp Stage 2+3 thành **một classifier multi-class fine-tuned (SetFit-style)**, nhận input đã fuse [tag text ⊕ agent-domain text], xuất xác suất calibrated cho cả 3 class, và áp dụng **một** ngưỡng escalate duy nhất chọn theo risk-coverage curve (Chow's reject rule) — thay cho 2 threshold tay tách biệt.

---

## 2. Bối cảnh bài toán


| Thuộc tính       | Giá trị                                                                           |
| ---------------- | --------------------------------------------------------------------------------- |
| Input            | `tag1`, `tag2` (1-10 từ), `scale` (binary/star5/star10/pct100/unbounded), `value` |
| Context tùy chọn | agent description, OASF domains/skills, services                                  |
| Output           | 3 class: `junk` / `quality` / `quantity`                                          |
| Phân phối class  | lệch mạnh — quality ~~70%, quantity ~30%, junk hiếm (~~1-3%)                      |
| Label set        | nhỏ — gold (~~320 human) + silver (~~1689 weak-supervision)                       |
| Yêu cầu          | high-throughput, latency thấp, LLM là fallback đắt                                |


Đây là bài toán **few-shot multi-class short-text classification với side-information** — không có gì đặc thù blockchain đòi hỏi kiến trúc khác thường. Literature đã có giải pháp validated cho đúng shape này.

---

## 3. Đánh giá pipeline hiện tại (Run 13)

### 3.1 Điểm đúng — nên giữ

- **Cascade rule → ML → LLM**: rule (Go) lọc phần dễ trước khi tốn tài nguyên ML/LLM. Đúng nguyên lý cost-aware cascade.
- **Reject option / selective classification**: chỉ trả kết quả khi đủ tin cậy, escalate phần còn lại cho oracle đắt hơn (LLM). Đây là kỹ thuật ML đã được peer-review (Chow 1970; Cortes et al., *Learning with Rejection*; Geifman & El-Yaniv, *SelectiveNet*, ICML). Nguyên lý này **đúng và nên giữ**.
- **Audit thực nghiệm trước khi quyết định**: Run 7/10 đo được symmetric tie-break cho accuracy < random (42.7%, 25.9%) → quyết định bỏ. Đây là good practice, không phải vấn đề.

### 3.2 Điểm cần sửa — gốc rễ

**(a) Frozen embedding ép phải dùng luật bất đối xứng**

`benchmarks/per_tag_svm.py::train_bge_quality_gate()` tự ghi nhận lý do giới hạn SVM một chiều:

> *"low confidence reflects unfamiliar business/service-domain vocabulary, not evidence of a metric"*

Đây chính là dấu hiệu embedding **chưa từng học domain vocabulary cho phía "quantity"**. Research xác nhận: *"frozen sentence embeddings... performance limited compared to standard full fine-tuning"*, trong khi SetFit (fine-tune contrastive trước khi train classifier head) đạt **92.7% accuracy với 8 example/class** trên benchmark chuẩn (IMDB) — vượt xa linear-probe-trên-frozen-embedding. Sửa đúng gốc rễ (fine-tune) sẽ cho phép classifier tự tin & chính xác ở **cả hai chiều**, không cần luật bất đối xứng.

**(b) Domain context bị threshold tay tách rời, không phải input của model**

Stage 3 cosine (`THRESH_IN_DOMAIN=0.55`) và Stage 2 SVM (`SVM_QUALITY_THRESH=0.80`) là hai ngưỡng **tune riêng lẻ**, không tối ưu đồng thời. Cách chuẩn hơn: agent-domain text là **một phần input** của cùng một classifier (late fusion), để model tự học cách weight 2 nguồn signal — không cần 2 threshold rời rạc.

Đáng chú ý: kiến trúc late-fusion này **đã tồn tại trong codebase** (`shared/linear_classifier.py::EnrichedLinearClassifier`, route `model=linear_enriched`) nhưng không được chọn làm production path.

---

## 4. Thiết kế đề xuất

```
Stage 0-1 (Go, KHÔNG ĐỔI)
  self-feedback gate → rule cascade (junk/quantity/quality)
            │
            ▼ "others"
┌──────────────────────────────────────────────────────────┐
│ Stage 2 — Unified Fine-tuned Classifier (THIẾT KẾ MỚI)   │
│                                                            │
│  Encoder: bge-small fine-tuned contrastive (SetFit-style) │
│           trên agent_enriched (group_a + group_b)         │
│  Input:   fuse([tag1, tag2, scale] ⊕ [agent_domain_text]) │
│  Output:  P(junk), P(quality), P(quantity) — calibrated   │
│           (Platt scaling / temperature scaling)            │
└──────────────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3 — Calibrated Reject Rule (Chow's rule)            │
│                                                            │
│  τ chọn qua risk-coverage curve trên validation set        │
│  (1 NGƯỠNG DUY NHẤT, áp dụng đều cho mọi class)            │
│                                                            │
│  max(P) ≥ τ  → trả kết quả (argmax class)                 │
│  max(P) < τ  → escalate Stage 4                            │
└──────────────────────────────────────────────────────────┘
            │
            ▼ escalate
Stage 4 (Python, KHÔNG ĐỔI)
  LLM — V8 prompt, scale-conditioned
```

### 4.1 Vì sao gộp Stage 2+3 thành một model


|                           | Cascade tay hiện tại                                                 | Unified + calibrated reject                                          |
| ------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Số threshold cần tune     | 2, tune riêng lẻ (grid search ad hoc)                                | 1, tune qua risk-coverage curve (tối ưu trực tiếp coverage/accuracy) |
| Gốc rễ embedding yếu      | Vá bằng luật bất đối xứng                                            | Sửa trực tiếp bằng fine-tune contrastive                             |
| Domain context            | Stage riêng, hard threshold                                          | Input feature, model tự học weight                                   |
| Đối xứng giữa class       | Không — chỉ được khẳng định "quality"                                | Có — escalate rule áp dụng đều cho mọi class                         |
| Khi data tăng             | Phải re-audit cascade (lặp lại Run 7→13)                             | Retrain + recalibrate threshold tự động qua risk-coverage curve      |
| Số stage / nhánh đặc biệt | 4 stage, nhiều nhánh (unbounded-safe, in-domain-bounded-low-conf...) | 2 stage, logic đồng nhất                                             |
| Cơ sở lý thuyết           | Ad hoc, derived từ audit riêng (đúng cho data này, không tổng quát)  | Chow's rule + SetFit — published, validated, tổng quát               |


### 4.2 Quy tắc reject cụ thể (Chow's rule)

Thay vì hard-code `if quality_prob > 0.80 and not unbounded`, chọn τ bằng cách:

1. Tính `max(P)` cho mọi record trong validation set.
2. Sắp theo `max(P)` giảm dần, tính accuracy tích lũy (risk) tại mỗi mức coverage.
3. Vẽ risk-coverage curve → chọn τ tại điểm coverage mong muốn (ví dụ: risk ≤ 5% lỗi trên phần được giữ lại).
4. τ này tự động re-tune mỗi khi retrain — không cần lặp lại quy trình audit Run 7→13 bằng tay.

### 4.3 Input fusion cụ thể

```python
def build_fused_text(tag1: str, tag2: str, scale: str, agent_domain_text: str) -> str:
    """Một câu duy nhất cho contrastive fine-tuning — không tách 2 tower."""
    tag_part = f"{tag1} {tag2} {scale}".strip()
    if agent_domain_text:
        return f"{tag_part} [SEP] {agent_domain_text}"
    return tag_part
```

Fine-tune SetFit trên câu fused này (positive pairs = cùng class, negative pairs = khác class), sau đó train classifier head 3-class (không phải binary one-directional).

---

## 5. Kế hoạch triển khai


| Bước | Việc                                                                                                                                                       | File                                             | Cần data/model mới?                                    |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| 1    | Cài `setfit` (hoặc tự code contrastive loop bằng `sentence-transformers` losses: `MultipleNegativesRankingLoss` / `BatchHardTripletLoss`)                  | `pyproject.toml`, `requirements.txt`             | Không                                                  |
| 2    | Viết `benchmarks/train_unified_classifier.py`: load `group_a.parquet` + `group_b.parquet`, build fused text, fine-tune encoder + train 3-class head        | mới                                              | Cần `data/splits/agent_enriched/*.parquet` (đã có sẵn) |
| 3    | Viết script tính risk-coverage curve trên validation split, xuất τ tối ưu theo coverage mục tiêu                                                           | `benchmarks/calibrate_reject_threshold.py` (mới) | Không                                                  |
| 4    | Viết `shared/unified_classifier.py`: load model đã fine-tune, expose `classify(tag1, tag2, scale, agent_domain_text) -> (category, prob, confident: bool)` | mới                                              | Artifact từ bước 2                                     |
| 5    | Sửa `shared/three_tier.py` (hoặc thêm route mới `model=unified`) để gọi `unified_classifier` thay Stage 2+3, giữ nguyên Stage 4 LLM fallback               | `shared/three_tier.py` hoặc file mới song song   | Không                                                  |
| 6    | Re-run `benchmarks/eval_pure_others.py`-style evaluation trên CHÍNH module mới để so sánh Macro F1 / coverage / LLM-call-rate với Run 13                   | sửa hoặc bản mới của `eval_pure_others.py`       | Không                                                  |
| 7    | A/B hoặc shadow-traffic so sánh trước khi chuyển production traffic                                                                                        | ops, ngoài repo                                  | Không                                                  |


**Lưu ý quan trọng**: bước 6 bắt buộc — phải đo trên cùng gold set (`data/labelled/pure_others_to_label.csv`) trước khi quyết định chuyển production, để có số liệu so sánh thật giữa thiết kế cũ và mới, không chỉ dựa trên lý thuyết.

---

## 6. Rủi ro & đánh đổi cần thừa nhận

- **Cần chạy lại toàn bộ training/calibration** — không phải sửa nhỏ. Thời gian thực tế: fine-tune SetFit trên vài nghìn record thường mất vài phút trên CPU/M-series, nhưng cần thời gian viết + test script mới.
- **Risk-coverage threshold cần đủ validation data để ổn định** — với gold set hiện tại (~320 human-labeled), curve có thể nhiễu ở phần coverage cao; nên cross-validate hoặc bootstrap để τ không overfit vào một split cụ thể.
- **Một model 3-class thay 2 stage** đơn giản hóa code, nhưng mất đi tính "explainable" của từng nhánh riêng (ví dụ rule "unbounded → quantity an toàn vì cấu trúc" hiện đang tách biệt rõ trong code). Nên giữ rule cấu trúc đó (unbounded/value<0 → loại trừ quality) làm **post-filter sau khi có P(.)**, không để model tự học lại điều đã biết chắc 100% — tức là: áp constraint cứng trước, rồi mới renormalize xác suất 3 class trên phần còn hợp lệ, rồi mới áp reject rule.
- **Không nên bỏ hoàn toàn audit Run 7→13** — số liệu đo được ở đó (ví dụ: 42.7% accuracy khi assert quantity bằng frozen embedding) là baseline tốt để so sánh "trước/sau" khi đánh giá thiết kế mới có thực sự khắc phục được vấn đề gốc hay không.

---

## 7. Tham khảo

- Chow, C.K. — lý thuyết error-reject trade-off gốc (1970).
- Cortes, DeSalvo, Mohri — *Learning with Rejection*.
- Geifman & El-Yaniv — *Selective Classification for Deep Neural Networks* (ICML), SelectiveNet.
- HuggingFace — *SetFit: Efficient Few-Shot Learning Without Prompts*.
- *SetFit ModernBERT* benchmark — 92.7% accuracy với 8 example/class trên IMDB.
- *Fine-Tuning Causal LLMs for Text Classification: Embedding-Based vs. Instruction-Based Approaches* (arxiv 2512.12677).

