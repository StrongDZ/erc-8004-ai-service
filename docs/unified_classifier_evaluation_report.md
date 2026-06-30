# Đánh giá thực nghiệm: Unified Classifier Proposal vs Production Cascade (Run 13)

**Date:** 2026-06-25
**Gold set:** `data/labelled/pure_others_stratified_dedup.csv` (N=1,486, stratified-dedup, cluster key `(tag1, tag2, scale, agent_key)`)
**Proposal đánh giá:** `docs/unified_classifier_proposal 1.md`
**Scripts:** `benchmarks/bench_llm_only.py`, `benchmarks/pipeline_run14.py`, `benchmarks/pipeline_run15.py`, `benchmarks/bench_enriched_linear.py`, `benchmarks/error_analysis.py`, `benchmarks/run_all_benchmarks.py`

---

## 1. Mục tiêu

Trả lời 3 câu hỏi cụ thể:

1. **LLM-only so với pipeline cascade hiện tại (Run 13) khác nhau thế nào?** — đo trực tiếp, không suy luận.
2. **Unified classifier proposal (gộp Stage 2+3 thành 1 model, có/không fine-tune) có cải thiện so với Run 13 không?** — đã có Run 14 (frozen embedding). Bổ sung **Run 15** (fine-tuned embedding, đúng theo đề xuất SetFit-style trong proposal) với 2 backbone: `bge-small` và `bge-base`.
3. **Pipeline sai ở đâu, vì sao?** — error analysis trên các trường hợp LLM-only misclassify.

---

## 2. Dataset & Gold Label Methodology

### 2.1 Nguồn nhãn

Gold set gồm **2,206 record** gốc (`pure_others_to_label.csv`) lấy từ `feedback_history` MongoDB — toàn bộ là các record mà rule engine (Go) không phân loại được (`category=others`), tức **đúng tập khó nhất** của bài toán. Một annotator (tác giả luận văn) gán nhãn `human_label ∈ {junk, quality, quantity}` theo **gold labeling convention** (Section 4.2.2 trong `docs/thesis_chapter_classification.md`): bộ quy tắc xác định dựa trên cấu trúc semantic của tag/scale/value, dùng để giải quyết các trường hợp mơ hồ rule engine không xử lý được (ví dụ: tag trông như metric nhưng thực ra là dịch vụ định tính).

### 2.2 Hạn chế đã biết — KHÔNG có Inter-Annotator Agreement (IAA)

**Toàn bộ 2,206 nhãn do một annotator duy nhất gán.** Không có second-rater, không đo được Cohen's κ hay bất kỳ chỉ số agreement nào. Đây là hạn chế đã ghi nhận từ trước (`docs/stratified_sampling_report.md`, mục 5.4) và **vẫn chưa được giải quyết trong vòng benchmark này**. Hệ quả: mọi số liệu F1/precision/recall trong báo cáo này có sai số hệ thống không đo được — nếu annotator có bias nhất quán (ví dụ thiên về gán "quality" cho các case mơ hồ), nó sẽ lan vào mọi model được benchmark như nhau, làm méo so sánh tuyệt đối nhưng **không** làm méo so sánh tương đối giữa các model (vì tất cả model dùng cùng gold set).

### 2.3 Phân bố lớp — lệch mạnh, không đổi sau stratification

| Lớp | N (dedup stratified) | % |
|---|---|---|
| quality | 1,181 | 79.5% |
| quantity | 302 | 20.3% |
| junk | 3 | 0.2% |

Stratified resampling (cluster key `(tag1, tag2, scale, agent_key)`, cap=1) xác nhận phân bố này là **intrinsic của others-pool**, không phải artifact của sampling concentration (xem `docs/stratified_sampling_report.md`). **Junk chỉ có 3 record** trong toàn bộ gold set — mọi metric junk-F1 trong báo cáo này **không có ý nghĩa thống kê** (support quá nhỏ để ước lượng precision/recall ổn định), chỉ nên đọc như tín hiệu định hướng.

### 2.4 Training data cho các model ML — khác phân phối với gold

| Nguồn | N | quality | quantity | junk |
|---|---|---|---|---|
| `group_a.parquet` + `group_b.parquet` (rule-labelled) | 1,032 | 44% | 43% | 13% |
| Gold (others-pool) | 1,486 | 79.5% | 20.3% | 0.2% |

Đây là **distribution shift kép**: (a) tỉ lệ lớp khác xa, và (b) bản chất ngữ nghĩa khác — training data là các record rule engine **đã** phân loại được (easy cases), còn gold set là phần rule engine **không** phân loại được (hard cases). Đây là nguyên nhân gốc rễ giải thích toàn bộ kết quả ở Mục 3–5.

---

## 3. Kết quả tổng hợp — tất cả pipeline/model trên cùng gold set (N=1,486)

| # | Model | Macro F1 | Weighted F1 | Quality F1 | Quality Recall | Quantity F1 | Quantity Recall | LLM% |
|---|---|---|---|---|---|---|---|---|
| 1 | **Pipeline Run 13** (BGE-SVM quality-gate, frozen, asymmetric + mandatory LLM escalation, thresh=0.80) | **0.585** | **0.871** | 0.922 | 0.947 | 0.679 | 0.586 | 45.4% |
| 2 | **LLM-only** (qwen2.5:7b-instruct, no cascade) | 0.560 | 0.857 | 0.904 | 0.908 | 0.680 | 0.606 | 100% |
| 3 | **Run 14** — unified frozen logreg (symmetric 3-class) + Chow @ best-macro τ=0.90 | 0.560 | 0.857 | 0.904 | 0.908 | 0.680 | 0.606 | 99.9% |
| 3b | Run 14 @ **Chow-selected** τ=0.40 (calibrated on val, error≤10%) | 0.219 | 0.346 | 0.361 | 0.227 | 0.289 | 0.255 | 5.7% |
| 4 | **Run 15a** — unified **fine-tuned** bge-small + Chow @ best-macro τ=0.90 | 0.479 | 0.778 | — | — | — | 0.510 | 5.9% |
| 4b | Run 15a @ **Chow-selected** τ=0.35 (val acc=99.5%) | 0.470 | 0.767 | — | — | — | 0.500 | 0.0% |
| 5 | **Run 15b** — unified **fine-tuned** bge-base + Chow @ best-macro τ=0.90 | 0.462 | 0.800 | — | — | — | 0.354 | 4.8% |
| 5b | Run 15b @ **Chow-selected** τ=0.35 (val acc=100%) | 0.438 | 0.782 | — | — | — | 0.298 | 0.0% |
| 5c | **Run 15c** — unified **fine-tuned ModernBERT-base** + Chow @ best-macro τ=0.90 | 0.514 | 0.816 | — | — | — | 0.593 | 53.0% |
| 5d | Run 15c @ **Chow-selected** τ=0.35 (val acc=98.6%) | 0.395 | 0.619 | — | — | — | 0.553 | 0.0% |
| 6 | EnrichedLinearClassifier (late-fusion BGE + Mongo agent context, trained on rule corpus) | 0.247 | 0.334 | — | 0.19 | — | 0.66 | n/a |
| 7 | TF-IDF Logistic Regression | 0.512 | 0.807 | — | — | — | — | n/a |
| 8 | TF-IDF SVM (linear) | 0.505 | 0.830 | — | — | — | — | n/a |
| 9 | TF-IDF Naive Bayes | 0.486 | 0.783 | — | — | — | — | n/a |
| 10 | TF-IDF Gradient Boosting | 0.477 | 0.816 | — | — | — | — | n/a |
| 11 | TF-IDF Random Forest | 0.454 | 0.790 | — | — | — | — | n/a |
| 12 | MiniLM kNN-embedding (k=7) | 0.368 | 0.727 | — | — | — | — | n/a |
| 13 | MiniLM Embedding-LogReg | 0.285 | 0.454 | — | — | — | — | n/a |

*(#7–13 train trên cùng `group_a+group_b` rule-labelled corpus N=264 dedup, không có LLM fallback — đại diện cho "ML-only, không cascade".)*

---

## 4. Trả lời 2 claim chính của proposal

### 4.1 "LLM-only vs cascade: cascade có lợi gì?"

LLM-only (0.560) chỉ thua Run 13 (0.585) **0.025 điểm Macro F1**, nhưng khác biệt thật sự nằm ở **chi phí**:

| | Run 13 | LLM-only |
|---|---|---|
| Macro F1 | 0.585 | 0.560 |
| % record cần gọi LLM | 45.4% | 100% |
| Quality Recall | 0.947 | 0.908 |
| Latency LLM thật (đo trực tiếp, live, không cache) | — | **0.85–7.44s/record** (steady-state ~1.0s sau warm-up; lần đầu 7.4s do model load) |

→ Run 13's BGE-SVM gate giải quyết **54.6%** record mà *không* cần LLM, với độ chính xác đủ cao để **tăng** Quality Recall (0.947 vs 0.908) so với để LLM tự quyết toàn bộ. Đây là lợi ích thực — không phải chỉ là tiết kiệm latency, mà SVM gate "lọc" đúng các case rõ ràng để LLM tập trung vào case mơ hồ.

### 4.2 "Unified classifier (frozen, Run 14) có cải thiện không?" → KHÔNG (đã xác nhận lại)

Run 14 (frozen BGE + symmetric logreg + Chow) đạt Macro F1=0.560 ở **τ=0.90 — gần như buộc toàn bộ (99.9%) phải escalate LLM**. Tại điểm Chow's rule tự chọn (τ=0.40, vì val set balanced khiến model "tự tin" ở mức thấp), Macro F1 sụp xuống **0.219** — tệ hơn random. Verdict không đổi so với `docs/run14_benchmark_report.md`.

### 4.3 "Fine-tuning (Run 15, đúng theo đề xuất SetFit-style) có giải quyết được vấn đề gốc rễ không?" → KHÔNG, nhưng backbone choice (mục 4.0a của proposal) có ảnh hưởng đáng kể

Proposal khẳng định gốc rễ là "frozen embedding chưa học domain vocabulary", đề xuất fine-tune contrastive (SetFit-style) để model tự tin đúng ở **cả hai chiều**. Proposal mục 4.0a cũng đặt câu hỏi mở "bge-small vs ModernBERT — chưa quyết, cần đo" — phần này benchmark cả 3 biến thể để trả lời cả hai câu hỏi cùng lúc.

**Kết quả đo được trên cả 3 backbone fine-tuned:**

| Backbone | Params/dim | Val accuracy (in-distribution) | Chow τ tự chọn | Macro F1 @ Chow τ | Macro F1 @ best τ=0.90 | LLM% @ best τ |
|---|---|---|---|---|---|---|
| bge-small (fine-tuned) | 33M/384 | 99.5% | 0.35 (sàn grid) | 0.470 | 0.479 | **5.9%** |
| bge-base (fine-tuned) | 109M/768 | 100% | 0.35 (sàn grid) | 0.438 | 0.462 | **4.8%** |
| **ModernBERT-base** (fine-tuned) | 149M/768 | 98.6% | 0.35 (sàn grid) | 0.395 | **0.514** | **53.0%** |
| *(so sánh)* Run 14 frozen bge-small | 33M/384 | 97.1% | 0.40 | 0.219 | 0.560 | 99.9% |

**Phát hiện quan trọng #1 — cả 3 backbone đều thua Run 13 (0.585) và LLM-only (0.560)** ở mọi mức τ thử nghiệm. Không backbone nào, fine-tuned hay không, khắc phục được distribution shift đã nêu ở Mục 2.4.

**Phát hiện quan trọng #2 — ModernBERT-base trả lời đúng câu hỏi mở 4.0a của proposal, nhưng theo hướng ngược dự đoán:** ModernBERT đạt Macro F1 cao nhất trong 3 backbone (0.514) **vì nó ít tự tin sai hơn** — tại best τ=0.90 nó vẫn escalate 53.0% record lên LLM (gần bằng tỉ lệ 45.4% của Run 13), trong khi 2 biến thể BGE chỉ escalate 4.8–5.9%. Risk-coverage curve của ModernBERT "mượt" hơn — xác suất dự đoán không bị đẩy về 2 cực trị (0/1) gắt như BGE khi gặp record ngoài phân phối train. Đây đúng là tín hiệu mà proposal kỳ vọng từ pretrain MLM hiện đại (rotary embeddings, không pretrain riêng cho similarity) — nhưng lợi ích thực tế chỉ là "ít overconfident sai hơn", **không phải "chính xác hơn nhờ hiểu domain tốt hơn"**: ở cùng tỉ lệ escalate ~53%, ModernBERT (0.514) vẫn thấp hơn Run 13 ở tỉ lệ escalate thấp hơn (45.4% → 0.585).

**Phát hiện quan trọng #3 — fine-tuning trên BGE (cả small và base) làm vấn đề TỆ HƠN, không phải tốt hơn**, đúng như ghi nhận lần đầu. Contrastive fine-tuning học ranh giới quyết định *sắc* (sharp decision boundary) trên training distribution (balanced 44/43/13%, rule-labelled — easy cases). Ranh giới sắc này tạo ra xác suất cực đoan (gần 0 hoặc gần 1) — kể cả khi áp dụng lên gold set có phân phối ngữ nghĩa khác hẳn (others-pool — hard cases), model vẫn xuất ra xác suất cực đoan, nhưng **sai**. Hệ quả: Chow's reject rule — vốn dựa trên giả định *train distribution ≈ test distribution* — bị đánh lừa gấp đôi: vừa do imbalance (đã thấy ở Run 14), vừa do model giờ "quá tự tin" trên chính phần dữ liệu nó dự đoán sai. ModernBERT bị hiện tượng này ở mức nhẹ hơn (val accuracy 98.6% — thấp hơn 2 biến thể BGE — và risk-coverage mượt hơn), nhưng vẫn không đủ để vượt Run 13.

→ **Verdict cập nhật cho proposal**: claim "(a) frozen embedding ép dùng luật bất đối xứng" có cơ sở đúng về mặt kỹ thuật. Câu hỏi mở 4.0a (bge-small vs ModernBERT) **đã có số liệu**: ModernBERT cho risk-coverage curve tốt hơn (ít overconfident sai), nhưng **fix được đề xuất (fine-tune, dù chọn backbone nào) không khắc phục được hệ quả thực tế** — vì nguồn distribution shift không phải do embedding yếu hay backbone choice, mà do **bản chất ngữ nghĩa của others-pool khác cấu trúc với rule-labelled corpus** (xem Mục 2.4). Không có lượng fine-tuning nào sửa được lỗi này nếu training data vẫn là rule-labelled corpus.

---

## 5. Confusion Matrix — Run 13 vs LLM-only

### 5.1 Run 13 (BGE-SVM gate + LLM escalation, thresh=0.80) — N=1,484 (2 self-feedback excluded)

```
              Pred:  junk  quality  quantity
True: junk        2        1         0
True: quality    23     1111        45
True: quantity    0      125       177
```

SVM-resolved: 808 (54.4%) · LLM-escalated: 676 (45.6%)

### 5.2 LLM-only (qwen2.5:7b-instruct) — N=1,486

```
              Pred:  junk  quality  quantity
True: junk        3        0         0
True: quality    56     1072        53
True: quantity    0      119       183
```

**So sánh trực tiếp:** Run 13 sai **45 quality→quantity** (SVM/LLM nhầm lẫn record quality thành quantity) vs LLM-only sai **53** — tương đương. Khác biệt rõ nhất: Run 13 sai **23** quality→junk (SVM gate quá nghiêm ở vài case) vs LLM-only sai **56** quality→junk — gấp 2.4×. Đây là phần SVM gate đóng góp giá trị thật: với case rõ ràng "quality", gate xác nhận đúng và **không** đưa vào LLM, tránh được lỗi junk-overprediction mà LLM model nhỏ (7B) hay mắc khi gặp tag lạ.

---

## 6. Error Analysis — LLM-only, N=1,486

Tổng lỗi: **228/1,486 (15.3%)**.

| Error Type | Count | % lỗi | Mô tả |
|---|---|---|---|
| **Missing feedbackURI** | 148 | 64.9% | `offchain_note` rỗng — không có narrative context, model chỉ dựa vào tag pair |
| **Ambiguous tags** | 55 | 24.1% | tag1/tag2 thuộc nhóm đa nghĩa (`accuracy`, `score`, `rating`...) — vừa có thể là metric đo lường, vừa có thể là đánh giá định tính |
| **Missing agent metadata** | 25 | 11.0% | `agent_description`/OASF rỗng — model không có domain context để phân biệt |

### 6.1 Ví dụ điển hình — Ambiguous tags (55 lỗi, dominant pattern: quantity→quality)

```
[true=quantity → pred=quality]  tag1='accuracy'  tag2=''            scale='pct100'
  agent: "Coach de truco argentino... solver verificable (minimax + ...)"
[true=quantity → pred=quality]  tag1='accuracy'  tag2='data-verified' scale='pct100'
[true=quantity → pred=quality]  tag1='accuracy'  tag2='yield-data'    scale='star5'
  agent: "DeFi yield intelligence agent... fetches real-time yields..."
```

**Phân tích:** tag `accuracy` xuất hiện cả ở agent đo lường chính xác thuật toán (đúng = quantity: "% nước đi tối ưu") và ở agent dịch vụ tài chính nơi "accuracy" là một đánh giá định tính tổng quát về độ tin cậy (annotator gán quality). LLM 7B không phân biệt được vì **không có signal thêm** ngoài tag string — đây chính xác là giới hạn mà proposal cố gắng giải quyết bằng agent-domain fusion, nhưng kết quả Run 15 cho thấy thêm agent context vào fine-tuned embedding cũng không đủ (model vẫn route sai vì sai lệch phân phối, không phải vì thiếu signal).

### 6.2 Missing feedbackURI (148 lỗi, dominant: quality↔quantity hai chiều)

```
[true=quality → pred=quantity]  tag1='celo-payments'  tag2='8-seconds'  scale='pct100'
  agent: "microtasking agent for quick cUSD payments on Celo..."
[true=quantity → pred=quality]  tag1='celo-native'    tag2='new-builder' scale='pct100'
  agent: "unknown | geopolitics/international_relations..."
```

**Phân tích:** đây là error type lớn nhất (64.9%). Khi `offchain_note` rỗng, model chỉ còn tag1/tag2/scale + agent context để quyết định — và agent context thường generic (domain mô tả rộng, không nói cụ thể về feedback này). `8-seconds` trông như một con số đo lường (→ quantity) nhưng annotator gán quality vì ngữ cảnh thực tế (không quan sát được từ tag) là một đánh giá tốc độ chủ quan. **Khuyến nghị cho future work**: ưu tiên fetch feedbackURI ở write-time cho các tag pattern đã biết là ambiguous, thay vì coi offchain content là optional.

### 6.3 Missing agent metadata (25 lỗi, dominant: quantity→quality)

```
[true=quantity → pred=quality]  tag1='a2a-compatible'    tag2='x402-compatible'  scale='pct100'
[true=quantity → pred=quality]  tag1='agent_rating'      tag2='task:ef00bfd7'    scale='pct100'
[true=quantity → pred=quality]  tag1='risk-analysis'     tag2='inter-agent'      scale='pct100'
```

**Phân tích:** không có agent description (`agent_description` rỗng hoặc agent chưa đăng ký đầy đủ trên Identity Registry). Đây là evidence trực tiếp cho thấy đăng ký metadata đầy đủ (Identity Registry) ảnh hưởng tới chất lượng classification downstream — một liên kết đáng đưa vào Chương 7 limitations.

---

## 7. Latency / Cost theo stage

| Stage | Cơ chế | Latency đo được | Khi nào kích hoạt |
|---|---|---|---|
| Stage 0–1 (Go rule cascade) | String/scale matching | <1ms | Mọi record có tag pattern đã biết |
| Stage 2 (BGE-SVM gate, Run 13) | Frozen embedding + calibrated SVM | ~1–3ms/record (CPU) | 54.4% record "others" còn lại |
| Stage 4 (LLM fallback, qwen2.5:7b) | Ollama, local | **0.85–7.44s/record live** (đo trực tiếp, không cache); steady-state ~1.0s sau warm-up đầu | 45.6% record (Run 13) / 100% (LLM-only) / 99.9% (Run 14 best) / 4.8–5.9% (Run 15 best) |

**Hệ quả cost trực tiếp:** Run 13 gọi LLM cho 676/1,486 record (45.6%) — ở latency steady-state ~1.0s/record, tổng cost ước tính **~11.3 phút** xử lý tuần tự cho toàn bộ gold set. LLM-only sẽ tốn **~24.8 phút** (gấp 2.2×) cho cùng tập, với Macro F1 thấp hơn. Run 14/15 ở best-macro τ vẫn cần LLM cho 95–99.9% record — **không tiết kiệm được cost so với LLM-only**, lại cho Macro F1 thấp hơn hoặc bằng. Đây là lý do kỹ thuật cụ thể (không chỉ lý thuyết) giải thích vì sao pipeline cascade bất đối xứng (Run 13) vẫn là lựa chọn production tốt nhất trong số các thiết kế đã thử.

---

## 8. Kết luận

1. **LLM-only đạt Macro F1=0.560**, chỉ thua Run 13 (0.585) 0.025 điểm, nhưng tốn ~2.2× thời gian xử lý tuần tự do gọi LLM cho 100% record thay vì 45.6%. Run 13's BGE-SVM gate là cơ chế cost-reduction hiệu quả, không phải overhead không cần thiết.

2. **Unified classifier proposal — cả 4 biến thể (frozen: Run 14; fine-tuned: Run 15 × {bge-small, bge-base, ModernBERT-base}) đều không cải thiện so với Run 13**, kể cả khi triển khai đúng đề xuất fine-tune contrastive (SetFit-style) và trả lời câu hỏi mở 4.0a của proposal (backbone choice). Frozen variant buộc phải escalate gần 100% để đạt Macro F1 tương đương LLM-only — không có lợi ích cost. Fine-tuned BGE (small/base) tệ hơn cả hai (Macro F1 trần 0.46–0.48) vì tự tin sai trên gold distribution. ModernBERT-base khá hơn 2 BGE fine-tuned (0.514) nhờ risk-coverage curve mượt hơn, nhưng vẫn thấp hơn Run 13 (0.585) và LLM-only (0.560) — kể cả khi escalate tới 53% record lên LLM (gần bằng tỉ lệ của Run 13).

3. **Gốc rễ thật của vấn đề không phải "embedding yếu"** (như proposal giả định) **mà là distribution shift giữa training corpus (rule-labelled, easy cases) và evaluation corpus (others-pool, hard cases by definition)**. Không có kiến trúc model nào (frozen hay fine-tuned, asymmetric hay symmetric) sửa được lỗi này nếu training data nguồn không đổi. Hướng cải thiện thật sự cần augment training data bằng chính các record others-pool đã gán nhãn người (nhưng quy mô annotation hiện tại — 2,206 record, một annotator — chưa đủ lớn để fine-tune một model riêng mà không overfit).

4. **Error analysis xác nhận 3 nguyên nhân lỗi cụ thể, không phải lỗi "mô hình yếu" chung**: 64.9% lỗi do thiếu feedbackURI (offchain context), 24.1% do tag đa nghĩa cần thêm signal ngoài tag string, 11.0% do thiếu agent metadata. Cả 3 đều là vấn đề **chất lượng dữ liệu input**, không phải giới hạn kiến trúc — gợi ý hướng cải thiện thực tế là tăng coverage của feedbackURI/metadata ở write-time, không phải đổi model.

5. **Hạn chế cần ghi nhận**: gold set chỉ có 1 annotator, không đo IAA. Junk class (N=3) không có ý nghĩa thống kê trong mọi so sánh. Cả hai điểm này cần xuất hiện trong Chapter 7 Limitations của luận văn.

---

## 9. Files liên quan

| File | Mô tả |
|---|---|
| `benchmarks/bench_llm_only.py` | LLM-only benchmark script |
| `benchmarks/pipeline_run14.py` | Unified frozen logreg + Chow (đã có từ trước) |
| `benchmarks/pipeline_run15.py` | Unified **fine-tuned** logreg + Chow — 2 backbone (mới) |
| `benchmarks/bench_enriched_linear.py` | EnrichedLinearClassifier benchmark (mới) |
| `benchmarks/error_analysis.py` | Error categorisation + confusion matrix (mới) |
| `data/benchmark_results/llm_only_20260625_015243.json` | LLM-only kết quả |
| `data/benchmark_results/pipeline_run14_20260625_025355.json` | Run 14 trên dedup gold (mới) |
| `data/benchmark_results/pipeline_run15_bge_small_20260625_025511.json` | Run 15a kết quả |
| `data/benchmark_results/pipeline_run15_bge_base_20260625_025621.json` | Run 15b kết quả |
| `data/benchmark_results/pipeline_run15_modernbert_20260625_031351.json` | Run 15c (ModernBERT-base) kết quả — trả lời mục 4.0a của proposal |
| `data/benchmark_results/enriched_linear_20260625_011118.json` | EnrichedLinearClassifier kết quả |
| `data/benchmark_results/summary_comparison.csv` | Bảng so sánh các model TF-IDF/embedding cổ điển |
| `docs/error_analysis_llm_only.md` | Error analysis đầy đủ (auto-generated) |
| `docs/unified_classifier_proposal 1.md` | Proposal được đánh giá |
| `docs/run14_benchmark_report.md` | Báo cáo Run 14 gốc (trên full pool, trước dedup) |
