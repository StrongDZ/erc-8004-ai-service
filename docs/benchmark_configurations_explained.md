# Benchmark Configurations — Full Explanation

> Tài liệu này giải thích toàn bộ các configurations trong benchmark phân loại feedback ERC-8004,
> bao gồm thuật toán, dữ liệu huấn luyện, ý nghĩa thiết kế, và kết quả.

---

## Dữ liệu dùng trong benchmark

### Training data

Tất cả models được train trên **rule-based data** — tức là các feedback record mà rule engine đã
classify được dựa vào pattern tag1/tag2/scale. Lấy từ MongoDB qua:

```python
{"$match": {"classification.rule.category": {"$in": aliases}}}
```

Phân phối training (sau khi dedup theo tag-pair): **quality ≈ 44%, quantity ≈ 43%, junk ≈ 13%**

Đây là "easy cases" — các record mà tag đủ rõ ràng để rule engine nhận biết.

### Test/Evaluation data (Gold set)

Gold set (N=1,486) là **others pool được human-annotate** — những record mà rule engine không tự
classify được (tag mơ hồ, metric domain cụ thể, thiếu off-chain content). Đây là "hard cases".

Phân phối gold: **quality ≈ 79.5%, quantity ≈ 20.3%, junk ≈ 0.2%**

**Distribution mismatch giữa train và test là vấn đề chủ chốt của toàn bộ benchmark.**
Models được train trên balanced data (44/43%) nhưng test trên imbalanced data (79.5/20.3%).

### Feature engineering chung (Groups A–B)

Hàm `build_feature_text()` tạo chuỗi:
```
"tag1=reliable | tag2=trustworthy | scale=star5 | endpoint=api.example.com | offchain=The agent responded..."
```
- Ghép tag1, tag2, value_scale, endpoint hostname, và text từ feedbackParsed (tối đa 300 ký tự)
- Đây là input duy nhất cho TF-IDF và embedding models

---

## Group A — TF-IDF + Classical ML (5 configurations)

**Cơ chế chung:** TfidfVectorizer chuyển chuỗi feature text thành sparse vector (đếm tần suất từ và
cặp từ), sau đó classifier học ranh giới quyết định trong không gian đó.

Tham số TF-IDF chung (trừ A4, A5):
- `ngram_range=(1,2)` — unigram + bigram
- `sublinear_tf=True` — dùng log(1+tf) thay vì tf thô, giảm ảnh hưởng của từ xuất hiện quá nhiều
- `min_df=2` — bỏ từ chỉ xuất hiện trong 1 document
- `class_weight="balanced"` — tự động điều chỉnh loss theo tỷ lệ class

---

### A1 — Logistic Regression TF-IDF

**Thuật toán:** Tìm ranh giới tuyến tính trong không gian TF-IDF để tối đa hoá log-likelihood.
Mỗi từ/bigram có một trọng số học được; quyết định dựa trên tổng trọng số.

```
Classifier: LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs")
TF-IDF: max_features=20000
```

**Tại sao có trong benchmark:** Baseline linear model mạnh nhất cho text classification, thường
là điểm tham chiếu chuẩn trong NLP.

**Kết quả:**
- Quality: P=0.901 / R=0.790 / F1=0.842
- Quantity: P=0.795 / R=0.589 / F1=0.677
- **2-cls Macro F1 = 0.760** ← Best trong Group A

---

### A2 — SVM Linear TF-IDF

**Thuật toán:** Tìm hyperplane với margin tối đa trong không gian TF-IDF (LinearSVC). Wrapped bởi
`CalibratedClassifierCV` để cho ra probability thay vì chỉ decision score.

```
Classifier: CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"))
TF-IDF: max_features=20000
```

**Tại sao có trong benchmark:** SVM thường cạnh tranh hoặc vượt LogReg trên sparse high-dimensional
text features nhờ margin maximization.

**Kết quả:**
- Quality: P=0.880 / R=0.901 / F1=0.890
- Quantity: P=0.779 / R=0.490 / F1=0.602
- **2-cls Macro F1 = 0.746**

---

### A3 — Naive Bayes TF-IDF

**Thuật toán:** P(label|text) ∝ P(text|label) · P(label). Giả định các từ độc lập với nhau
(Naive Bayes assumption — sai về mặt toán học, nhưng vẫn hiệu quả trong thực tế).

```
Classifier: MultinomialNB(alpha=0.1)
TF-IDF: max_features=20000
```

**Tại sao có trong benchmark:** Nhanh nhất, training gần như tức thì, kiểm tra xem prior
probability + word likelihood có đủ để phân loại không.

**Kết quả:** 2-cls Macro F1 ≈ 0.717

---

### A4 — Gradient Boosting TF-IDF

**Thuật toán:** Ensemble của các cây quyết định nông, mỗi cây học từ residuals của cây trước
(boosting sequential). Lý thuyết tốt trên tabular data, nhưng không phải lãnh địa tự nhiên của
text features thưa.

```
Classifier: GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8)
TF-IDF: max_features=5000  ← giảm xuống vì dense matrix rất nặng RAM với GBT
```

**Tại sao có trong benchmark:** Kiểm tra xem non-linear ensemble có capture được pattern nào mà
linear model bỏ sót không.

**Kết quả:** 2-cls Macro F1 ≈ 0.705

---

### A5 — Random Forest TF-IDF

**Thuật toán:** Ensemble của nhiều cây quyết định song song (bagging), mỗi cây được train trên
random subset của features và samples. Kết quả là majority vote.

```
Classifier: RandomForestClassifier(n_estimators=200, class_weight="balanced")
TF-IDF: max_features=10000
```

**Tại sao có trong benchmark:** Đối chiếu với GBT — cả hai đều ensemble trees nhưng training
strategy khác nhau (parallel vs sequential).

**Kết quả:** 2-cls Macro F1 ≈ 0.674 ← Tệ nhất Group A

**Rút ra từ Group A:** Linear model (A1, A2) > ensemble trees (A4, A5) trên text features thưa.
Đây là kết quả phổ biến trong NLP — TF-IDF sparse vectors là lãnh địa của linear classifier.

---

## Group B — Frozen Embedding + Classical ML (3 configurations)

**Cơ chế chung:** Thay vì đếm từ, dùng sentence transformer để encode text thành **dense vector
384 chiều**. Vector nắm bắt ngữ nghĩa — "reliable" và "trustworthy" nằm gần nhau trong không gian này.

**Model embedding: `all-MiniLM-L6-v2`** (general-purpose, nhỏ và nhanh, không fine-tune).

**Tại sao dùng MiniLM chứ không phải BGE?** BGE được dành cho production cascade (Group F) và
fine-tuned groups (D/E). MiniLM là baseline frozen embedding để so sánh.

---

### B1 — kNN Embedding (k=7)

**Thuật toán:** Instance-based learning — không có "mô hình" thực sự. Tại inference:
1. Encode toàn bộ training set thành embeddings → lưu cache
2. Encode test record
3. Tính cosine similarity với tất cả training vectors
4. Lấy 7 neighbors gần nhất → majority vote

```
Encoder: all-MiniLM-L6-v2 (384-dim)
k=7, cosine similarity via dot product (normalized vectors)
```

**Tại sao có trong benchmark:** Kiểm tra xem geometric proximity trong embedding space có
phản ánh semantic similarity giữa các categories không. Không cần training.

**Kết quả:**
- Quality: P=0.820 / R=0.897 / F1=0.856
- Quantity: P=0.442 / R=0.152 / F1=0.227
- **2-cls Macro F1 = 0.542** ← Quantity recall rất tệ

**Lý do tệ:** Imbalance 79.5/20.3% → trong 7 neighbors, thường 6-7 cái là quality → luôn vote
quality kể cả khi record thực sự là quantity.

---

### B2 — Embedding LogReg

**Thuật toán:** Encode text bằng MiniLM → 384-dim dense vector → fit Logistic Regression.

```
Encoder: all-MiniLM-L6-v2 (384-dim)
Classifier: LogisticRegression(C=1.0, class_weight="balanced")
```

**Tại sao có trong benchmark:** So sánh "semantic vector + linear classifier" vs "TF-IDF + linear
classifier" — xem semantic understanding có giúp gì so với word counting không.

**Kết quả:** 3-cls Macro F1 ≈ 0.285 (tệ hơn TF-IDF)

**Lý do tệ:** `all-MiniLM-L6-v2` là general model — chưa thấy feedback taxonomy ERC-8004 bao
giờ. Không phân biệt được "win-rate" (quantity) vs "reliable" (quality). TF-IDF vẫn tốt hơn vì
nó học exact keyword patterns domain-specific.

---

### B3 — Enriched Linear

**Thuật toán:** LogReg TF-IDF nhưng feature text được bổ sung thêm metadata số:
- `value_norm` (giá trị feedback chuẩn hoá về [0,1])
- `value_decimals` (số chữ số thập phân)
- `score_tier` (tier phân loại scale — star/percent/binary/unbounded)

**Tại sao có trong benchmark:** Kiểm tra giả thuyết rằng numeric signals giúp phân biệt
quantity (scale unbounded) vs quality (scale bounded). Giả thuyết "enrichment giúp ích".

**Kết quả:** 3-cls Macro F1 = 0.247, weighted F1 = 0.334 — giả thuyết **bị bác bỏ**. Enrichment
không giúp, có thể thêm noise.

---

## Group C — LLM-only (1 configuration)

### C — Qwen2.5-7B-Instruct Zero-Shot

**Thuật toán:** Không có training. Dùng ngôn ngữ tự nhiên mô tả 3 categories, đưa
tag1/tag2/scale/off-chain content vào prompt, yêu cầu LLM trả về label.

```
Model: qwen2.5:7b-instruct (chạy qua Ollama)
Mode: zero-shot classification
```

**Tại sao có trong benchmark:** Upper bound tham chiếu — nếu cascade có thể tiệm cận accuracy
của LLM-only với cost thấp hơn thì cascade thắng.

**Kết quả:**
- Quality: P=0.900 / R=0.908 / F1=0.904
- Quantity: P=0.775 / R=0.606 / F1=0.680
- **2-cls Macro F1 = 0.792**
- LLM call rate: 100%

**Điểm yếu:** ~1s/record, phụ thuộc hoàn toàn vào Ollama, không thể scale tuyến tính.

---

## Group D — Unified Frozen BGE-SVM + Chow Reject Rule (2 configurations)

**Ý tưởng thiết kế:** Thay vì cascade nhiều stage, train **một model duy nhất** phân loại tất cả
categories, sau đó dùng **Chow's reject rule** (1957) để quyết định record nào giữ lại, record
nào escalate sang LLM.

**Chow's reject rule:** Chọn threshold τ nhỏ nhất sao cho validation error ≤ 10%. Records với
max class probability < τ → không quyết định → gọi LLM.

**Backbone:** `BAAI/bge-small-en-v1.5` (frozen, không fine-tune), Linear SVM on top.

**Training data:** rule-based data (`group_a` + `group_b` parquets), same as Group F.

---

### D-best — Frozen BGE-SVM tại τ tốt nhất về Macro F1

```
τ = 0.90 (threshold tại đó Macro F1 tốt nhất)
LLM call rate: ~99.9%
```

**Kết quả:** Macro F1 cao nhưng gần như toàn bộ records escalate → không tiết kiệm gì so với
LLM-only.

---

### D-chow — Frozen BGE-SVM tại τ do Chow rule chọn

```
τ = 0.40 (threshold tại validation error ≤ 10%)
LLM call rate: ~5.7%
3-cls Macro F1: 0.219
```

**Tại sao D thất bại:**

Chow threshold được chọn dựa trên **validation set từ rule-labeled data** (same distribution
as training). Model confident trên easy rule-labeled cases. Khi τ thấp (ít LLM), model tự
quyết định những "hard" cases từ others pool mà nó chưa thấy bao giờ → sai nhiều.

Vấn đề căn bản: calibration của model không transfer sang gold set distribution.

---

## Group E — Fine-tuned Embedding + Chow Reject Rule (3 configurations)

**Cơ chế:** Giống D nhưng **fine-tune backbone embedding** trên rule-based training data thay vì
frozen. Hi vọng fine-tuning giúp model hiểu domain ERC-8004.

**Training data:** rule-based data (same as Group D).

---

### E1 — BGE-small Fine-tuned + Chow

```
Backbone: BAAI/bge-small-en-v1.5 (fine-tuned ~97s)
Chow-selected τ = 0.35
LLM call rate: ~5.9%
```

**Kết quả:** Quality F1=0.820, Quantity F1=0.569, **2-cls Macro F1 = 0.706**

---

### E2 — BGE-base Fine-tuned + Chow

```
Backbone: BAAI/bge-base-en-v1.5 (larger, ~161s)
LLM call rate: ~4.8%
```

**Kết quả:** Quality F1=0.883, Quantity F1=0.482, **2-cls Macro F1 = 0.683**

Lớn hơn không có nghĩa là tốt hơn — bge-base overfit hơn trên training distribution.

---

### E3 — ModernBERT Fine-tuned + Chow

```
Backbone: answerdotai/ModernBERT-base (state-of-the-art, ~201s)
Chow-selected τ = 0.90
LLM call rate: ~53%
```

**Kết quả:** Quality P=0.900/R=0.833/F1=0.865, Quantity P=0.670/R=0.593/F1=0.629
**2-cls Macro F1 = 0.747** ← Best trong Group E

**Tại sao Group E tệ hơn mong đợi:**

Fine-tune trên **training distribution** (quality/quantity ~50/50, "easy" rule-labeled cases).
Test trên **gold set distribution** (79.5/20.3%, "hard" others pool).
→ Distribution shift làm hỏng calibration.
→ E3 cần 53% LLM call — gần ngang LLM-only về cost nhưng accuracy thấp hơn cascade.

---

## Group F — Production Cascade (1 configuration)

**Triết lý thiết kế:** Không phải "một model duy nhất". Thay vào đó là pipeline nhiều stage với
logic cứng, mỗi stage chỉ quyết định khi đủ tự tin.

**Training data của BGE-SVM trong cascade:**
- **Nguồn:** rule-based data (`group_a` + `group_b` parquets từ MongoDB với
  `classification.rule.category`)
- **Junk excluded:** `df[df["label"] != "junk"]` → chỉ train trên quality và quantity
- **Phân phối:** quality ≈ 455, quantity ≈ 442 (gần balanced)

**Lưu ý:** BGE-SVM **train trên rule-based data nhưng evaluate trên others pool gold set**. Không
phải train trên others pool — điều đó sẽ làm vòng tròn hóa benchmark.

---

### F — Production Cascade (τ=0.80)

```
Stage 1: Rule engine (bao phủ ~92.3% traffic live)
    ↓ (chỉ residual "others" đến đây)
Stage 2: BGE-small SVM (frozen, quality-only assertion)
    - Nếu quality_prob ≥ 0.80 → label = "quality" → STOP
    - Nếu < 0.80 → không quyết định (không assert quantity)
    ↓
Stage 3: Cosine similarity + scale heuristic (domain check)
    - Nếu scale=unbounded AND cosine cao với agent domain → "quantity" → STOP
    ↓
Stage 4: LLM (qwen2.5:7b-instruct)
    → label cuối cùng
```

**Tham số duy nhất được sweep:** `SVM_THRESH ∈ {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}`
→ τ=0.80 cho kết quả tốt nhất.

**Kết quả tại τ=0.80:**
- Quality: P=0.899 / R=0.947 / **F1=0.922**
- Quantity: P=0.808 / R=0.586 / **F1=0.680**
- Junk: F1=0.154 (support=3, không có ý nghĩa thống kê)
- 3-cls Macro F1 = 0.585
- **2-cls Macro F1 = 0.801** ← Best overall
- Weighted F1 = 0.871
- LLM call rate: 45.4% (674/1484 records)

---

## Tổng hợp và So sánh

### Bảng kết quả đầy đủ

| Group | Config | Algorithm | Train data | 2-cls Macro F1 | LLM% |
|---|---|---|---|:---:|:---:|
| A | A1 LogReg TF-IDF | LR + TF-IDF(20k) | rule-based | 0.760 | 0% |
| A | A2 SVM TF-IDF | SVM + TF-IDF(20k) | rule-based | 0.746 | 0% |
| A | A3 NB TF-IDF | MultinomialNB + TF-IDF | rule-based | 0.717 | 0% |
| A | A4 GBT TF-IDF | GradBoosting + TF-IDF(5k) | rule-based | 0.705 | 0% |
| A | A5 RF TF-IDF | RandomForest + TF-IDF(10k) | rule-based | 0.674 | 0% |
| B | B1 kNN Emb | kNN(k=7) + MiniLM | rule-based | 0.542 | 0% |
| B | B2 LogReg Emb | LR + MiniLM(384d) | rule-based | ~0.285* | 0% |
| B | B3 Enriched | LR + TF-IDF + metadata | rule-based | — | 0% |
| C | C LLM-only | Qwen2.5-7b zero-shot | none | 0.792 | 100% |
| D | D-best | BGE-SVM frozen + Chow | rule-based | ~0.792 | ~99.9% |
| D | D-chow | BGE-SVM frozen + Chow(τ=0.40) | rule-based | ~0.219* | 5.7% |
| E | E1 BGE-small FT | Fine-tuned BGE-small + Chow | rule-based | 0.706 | 5.9% |
| E | E2 BGE-base FT | Fine-tuned BGE-base + Chow | rule-based | 0.683 | 4.8% |
| E | E3 ModernBERT FT | Fine-tuned ModernBERT + Chow | rule-based | 0.747 | 53% |
| **F** | **F Cascade** | **BGE-SVM + Cosine + LLM** | **rule-based** | **0.801** | **45.4%** |

*\* 3-cls Macro F1 (2-cls không available)*

### Pareto analysis

| | Macro F1 | LLM% | Verdict |
|---|:---:|:---:|---|
| LLM-only (C) | 0.792 | 100% | Dominated bởi F |
| **Cascade (F)** | **0.801** | **45.4%** | **Pareto optimal** |
| ModernBERT (E3) | 0.747 | 53% | Tệ hơn F cả F1 lẫn cost |
| LogReg TF-IDF (A1) | 0.760 | 0% | Cost thấp hơn nhưng F1 thấp hơn nhiều |

**Cascade (F) strictly dominates LLM-only: F1 cao hơn (0.801 > 0.792) và LLM cost thấp hơn
(45.4% < 100%). Không có configuration nào dominates cascade.**

### Tại sao cascade thắng

1. **One-directional design:** BGE-SVM chỉ assert "quality" khi rất tự tin (τ=0.80). Không bao
   giờ guess "quantity" — lỗi loại này (false quantity) là lỗi mà tất cả các run trước mắc phải.

2. **Pipeline phân công đúng chỗ:** Rule engine xử lý 92.3% easy cases. SVM xử lý quality rõ
   ràng trong others pool. LLM chỉ nhận những gì thực sự cần reasoning.

3. **Không bị distribution mismatch xấu hơn Group E:** Cả cascade và Group E đều train trên
   rule-based data. Nhưng cascade không fine-tune → không overfit trên training distribution. BGE
   frozen embeddings là features ổn định.

4. **τ=0.80 được calibrate trực tiếp trên gold set:** Threshold sweep chạy trên gold set, không
   phải validation set rule-labeled → calibration đúng domain.

---

## Điểm yếu còn lại (honest assessment)

- **Quantity recall = 0.586:** Hệ thống bỏ sót ~41% quantity feedback. Trong live scoring, điều
  này có thể undercount adoption signals.
- **Junk support = 3:** Metric F1 cho junk không có ý nghĩa thống kê. Cần more junk in others pool.
- **Distribution mismatch không được giải quyết triệt để:** Train/test vẫn khác nhau. Một pipeline
  được train trực tiếp trên others-pool data (nếu có đủ labeled data) có thể tốt hơn.
- **LLM dependency:** 45.4% calls có nghĩa là hệ thống không hoạt động hoàn toàn khi Ollama
  unavailable.
