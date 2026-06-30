# Benchmark Report — Pipeline Run 14: Unified Fused Logreg + Chow Reject Rule

**Date:** 2026-06-25  
**Script:** `benchmarks/pipeline_run14.py`  
**Gold pool:** `data/labelled/pure_others_to_label.csv` (N=2,199 after self-feedback exclusion)  
**Comparison baseline:** Pipeline Run 13 @ thresh=0.80  

---

## 1. Mục tiêu

Thử nghiệm thiết kế "unified classifier" từ `docs/unified_classifier_proposal.md`:
- Thay Stage 2 (SVM binary, one-directional) + Stage 3 (cosine domain check) bằng **một logreg 3-class duy nhất** trên vector fused `[tag+scale embedding ‖ agent_description embedding]`.
- Thay 2 threshold tay (`SVM_QUALITY_THRESH=0.80`, `THRESH_IN_DOMAIN=0.55`) bằng **một ngưỡng τ duy nhất** chọn theo Chow's reject rule.
- Giữ nguyên: Stage 0-1 rule cascade (Go), Stage 4 LLM fallback.

---

## 2. Thiết lập thực nghiệm

### 2.1 Training data

| Nguồn | N | Labels |
|---|---|---|
| `group_a.parquet` | 819 | quality=385, quantity=375, junk=59 |
| `group_b.parquet` | 213 | junk=76, quality=70, quantity=67 |
| **Tổng** | **1,032** | quality=455 (44%), quantity=442 (43%), junk=135 (13%) |

Training data **balanced hơn nhiều** so với gold pool (82% quality, 17% quantity, 0.1% junk). Đây là nguồn gốc của distribution shift.

Train/val split: 80/20 stratified (train=825, val=207).

### 2.2 Model

- **Encoder:** BGE-small-en-v1.5 (frozen, giống Run 13) — SetFit không có trong venv.
- **Input:** `fused_tag_text(tag1, tag2, value_scale)` → embed → concatenate với embed(agent_description[:500]).
- **Classifier head:** `LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs', max_iter=3000)`.
- **Dimension:** 2 × 384 = 768 (tag vec ‖ agent vec).
- **Structural constraint:** nếu `value_scale == "unbounded"` → zero P(quality) + renormalize trước argmax (giữ cùng heuristic với Run 13).

### 2.3 Chow reject threshold (τ)

Criterion: trên validation fold, tìm τ nhỏ nhất sao cho error rate trên phần được giữ lại ≤ 10%.

```
τ=0.40: coverage=1.00, error=0.029 ← qualifies → selected τ=0.40
```

Kết quả: τ=0.40 đã đạt error ≤ 10% ở coverage=100% → Chow chọn ngưỡng rất thấp (= giữ gần như tất cả). Giải thích: model rất confident trên val set (balanced, in-distribution), nhưng confident sai trên gold set (out-of-distribution, skewed labels).

---

## 3. Kết quả theo τ grid (N_gold=2,199, LLM fills escalations)

| τ | Macro F1 | Weighted F1 | Quality F1 | Qty F1 | Qty Recall | LLM% |
|---|---|---|---|---|---|---|
| 0.40 (Chow) | 0.1886 | 0.2950 | 0.303 | 0.259 | 0.245 | 5.7% |
| 0.45 | 0.2033 | 0.3549 | 0.384 | 0.222 | 0.198 | 14.9% |
| 0.50 | 0.2385 | 0.4277 | 0.467 | 0.243 | 0.204 | 26.2% |
| 0.55 | 0.3305 | 0.5488 | 0.580 | 0.406 | 0.334 | 42.9% |
| 0.60 | 0.3852 | 0.6363 | 0.671 | 0.478 | 0.394 | 55.4% |
| 0.65 | 0.4166 | 0.6863 | 0.723 | 0.518 | 0.444 | 65.7% |
| 0.70 | 0.4862 | 0.8065 | 0.855 | 0.585 | 0.514 | 85.9% |
| 0.75 | 0.5122 | 0.8306 | 0.873 | 0.637 | 0.582 | 91.4% |
| **0.80** | **0.5247** | **0.8412** | **0.881** | **0.661** | **0.614** | 93.5% |
| 0.85 | 0.5489 | 0.8675 | 0.913 | 0.660 | 0.614 | 98.3% |
| **0.90 (best)** | **0.5512** | **0.8683** | **0.914** | **0.657** | **0.614** | **99.9%** |

---

## 4. So sánh với Run 13 @ thresh=0.80

| Metric | Run 13 (thresh=0.80) | Run 14 (τ=0.80) | Run 14 (τ=0.90 best) | Δ vs Run 13 (best) |
|---|---|---|---|---|
| **Macro F1** | **0.566** | 0.525 | **0.551** | −0.015 |
| **Weighted F1** | **0.879** | 0.841 | **0.868** | −0.011 |
| Quality F1 | 0.927 | 0.881 | 0.914 | −0.013 |
| Quality Recall | **0.938** | 0.851 | 0.910 | −0.028 |
| **Quantity F1** | 0.660 | **0.661** | 0.657 | −0.003 |
| **Quantity Recall** | 0.598 | **0.614** | **0.614** | **+0.016** |
| LLM calls (%) | **53.4%** | 93.5% | **99.9%** | **+46.5pp** |

---

## 5. Phân tích nguyên nhân

### 5.1 Distribution shift là nguyên nhân chính

Model đạt **val accuracy = 0.971** trên validation fold (balanced, in-distribution) nhưng chỉ đạt Macro F1=0.551 trên gold (skewed 82% quality):

| Subset | Quality | Quantity | Junk |
|---|---|---|---|
| Training data | 44% | 43% | 13% |
| Gold pool | 82% | 17% | 0.1% |

`class_weight='balanced'` trong logreg bù lại training imbalance nhưng tạo ra **over-prediction của quantity/junk** khi test trên gold set skewed. Logreg không biết gold set có prior quality=82% — nó được train để predict 3 class đều nhau.

### 5.2 Tại sao LLM rate tăng vọt (99.9% ở best τ)

Classifier fused có confidence rất thấp trên gold records:
- Phần lớn records cần τ ≤ 0.40 để được giữ lại (coverage=100% ở τ=0.40 trên val).
- Nhưng trên gold, confidence thực tế phân tán — nhiều records bị xếp vào quantity với confidence cao nhưng sai (vì model bias từ balanced training).
- Để đạt Macro F1 ổn (0.55), pipeline phải đẩy phần lớn vào LLM — tức là classifier chỉ đóng vai filter, LLM làm phần lớn công việc.

Run 13 tránh được vấn đề này vì SVM binary (quality-vs-non-quality) không asserting "quantity" — chỉ asserting "quality" với ngưỡng cao, đẩy phần còn lại cho LLM. LLM rate 53.4% vs 99.9%.

### 5.3 Tại sao Chow τ=0.40 sai

Chow được calibrate trên val set (balanced) — error ≤ 10% ở τ=0.40 trên val. Nhưng trên gold (skewed), error tại τ=0.40 rất cao vì model incorrectly confident về quantity predictions. Chow's rule chỉ đúng khi **train distribution ≈ test distribution** — điều kiện này bị vi phạm ở đây.

---

## 6. Quan sát chính

**Quantity Recall tốt hơn Run 13 (+1.6pp)**: Model 3-class symmetric có thể assert "quantity" — Run 13 chỉ assert "quality" hoặc đẩy LLM. Kết quả: Run 14 bắt được thêm một phần quantity mà Run 13 miss.

**LLM cost tăng 2× (99.9% vs 53.4% ở best τ)**: Đây là đánh đổi lớn nhất. Nếu LLM = thứ đắt nhất trong pipeline, Run 14 tốn gấp đôi cost để đổi lấy Macro F1 thấp hơn (0.551 vs 0.566).

**Macro F1 thấp hơn (−0.015)**: Unified fused logreg thua Run 13's asymmetric cascade. Lý do: Run 13 khai thác được cấu trúc bài toán (quality = bounded + in-domain) bằng rule cứng; logreg phải học lại từ dữ liệu balanced ≠ test distribution.

---

## 7. Kết luận

Thiết kế "unified logreg" **không cải thiện** so với Run 13 trong điều kiện thực nghiệm này. Kết quả trả lời cụ thể cho từng claim của `unified_classifier_proposal.md`:

| Claim đề xuất | Thực tế đo được |
|---|---|
| "Fine-tune symmetric cho phép assert quantity chính xác" | Đúng một phần: qty recall tăng +1.6pp — nhưng đổi lấy LLM rate 99.9% |
| "Một τ duy nhất qua Chow thay 2 threshold tay" | Chow τ chỉ đúng khi train≈test distribution; bị miscalibrate do distribution shift |
| "Gộp Stage 2+3 đơn giản hóa pipeline" | Đúng về code — nhưng LLM cost tăng 2× nên trade-off không có lợi |
| "SetFit fine-tuning giải quyết embedding yếu" | SetFit không khả dụng trong venv; frozen BGE vẫn bị hạn chế |

**Verdict:** Run 13's asymmetric cascade khai thác cấu trúc bài toán (bounded → quality, unbounded → quantity, không biết → LLM) hiệu quả hơn một classifier đối xứng khi training distribution lệch xa test distribution. Nếu muốn cải thiện thật sự, cần giải quyết distribution shift: hoặc augment training data với nhiều records quality hơn (phản ánh đúng 82% prior), hoặc dùng SetFit thực sự để fine-tune encoder trên domain.

---

## 8. Files liên quan

| File | Mô tả |
|---|---|
| `benchmarks/pipeline_run14.py` | Script Run 14 |
| `data/benchmark_results/pipeline_run14_20260625_013440.json` | Kết quả đầy đủ |
| `data/benchmark_results/pipeline_run13_20260622_235831.json` | Run 13 baseline |
| `docs/unified_classifier_proposal.md` | Proposal được evaluate |

---

## 9. Appendix — Val set performance (in-distribution)

Trên val set (N=207, balanced labels), model đạt:

```
              precision  recall  f1-score  support
junk           0.96      1.00      0.98       27
quality        0.96      0.98      0.97       91
quantity       0.99      0.96      0.97       89
accuracy                           0.97      207
macro avg      0.97      0.98      0.97      207
```

Val accuracy 0.971 xác nhận model **không yếu** — vấn đề là distribution shift thuần túy, không phải năng lực mô hình.
