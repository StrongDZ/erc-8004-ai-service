# Stratified Sampling Robustness Check — Others-Pool Classifier

**Date:** 2026-06-24  
**Script:** `benchmarks/stratified_resample_gold.py`  
**Gold pool (source):** `data/labelled/pure_others_to_label.csv`  
**Subsample output:** `data/labelled/pure_others_stratified_cap5.csv`

---

## 1. Vấn đề cần giải quyết

Bộ gold-label gốc (N=2,206) bị **concentration bias** nghiêm trọng:

- Chain 8453 (Base) chiếm **68.4%** toàn bộ records
- Cluster `(chain=8453, tag1='tip', tag2='agent')` một mình có **284 records** — lớn hơn toàn bộ tập chain Ethereum (103 records)
- 659/728 cluster chỉ có ≤5 records, nhưng 69 cluster lớn kéo metric lên theo hướng thiên lệch

Nếu benchmark chỉ chạy trên full pool, kết quả F1 phản ánh performance trên distribution chain/tag-pair thực tế của hệ thống — không phải khả năng tổng quát hoá sang tag-pair ít gặp hơn.

---

## 2. Phương pháp stratified sampling

**Không** vẽ records mới hay yêu cầu gán nhãn thêm — toàn bộ records trong subsample đã có `human_label` từ đợt annotation gốc.

**Thuật toán:**
1. Định nghĩa cluster key: `(chain_id, tag1.lower(), tag2.lower())`
2. Với mỗi cluster: nếu `size ≤ cap` → giữ nguyên; nếu `size > cap` → random sample `cap` records (seed=42)
3. Cap mặc định: **5 records/cluster**

```python
for _, group in df.groupby("_cluster"):
    idx = list(group.index)
    kept_idx.extend(idx if len(idx) <= cap else rng.sample(idx, cap))
```

---

## 3. Thống kê so sánh

### 3.1 Full pool vs Stratified subsample

| Chỉ số | Full pool | Stratified (cap=5) |
|---|---|---|
| **N records** | 2,206 | 1,326 |
| **Số cluster** | 728 | 728 (toàn bộ) |
| **Tỉ lệ giữ lại** | 100% | 60.1% |
| Cluster bị cắt (size > 5) | — | 69/728 (9.5%) |
| Cluster giữ nguyên (size ≤ 5) | — | 659/728 (90.5%) |

### 3.2 Phân bố nhãn

| Label | Full pool N | Full pool % | Stratified N | Stratified % |
|---|---|---|---|---|
| quality | 1,820 | 82.5% | 1,082 | 81.6% |
| quantity | 383 | 17.4% | 241 | 18.2% |
| junk | 3 | 0.1% | 3 | 0.2% |

→ Phân bố nhãn **gần như không đổi** sau khi stratify — imbalance là intrinsic của tập dữ liệu, không phải do concentration bias.

### 3.3 Phân bố chain

| Chain | Full pool N (%) | Stratified N (%) | Δ |
|---|---|---|---|
| Base (8453) | 1,510 (68.4%) | 796 (60.0%) | −8.4pp |
| Celo (42220) | 528 (23.9%) | 374 (28.2%) | +4.3pp |
| Ethereum (1) | 103 (4.7%) | 103 (7.8%) | +3.1pp |
| BSC (56) | 49 (2.2%) | 38 (2.9%) | +0.7pp |
| Avalanche (43114) | 16 (0.7%) | 15 (1.1%) | +0.4pp |

→ Base chain giảm từ 68% → 60%. Các chain nhỏ hơn tăng tỉ trọng tương đối.

### 3.4 Top clusters bị cắt mạnh nhất

| Cluster (chain, tag1, tag2) | Label | Full pool | Stratified |
|---|---|---|---|
| 8453, 'tip', 'agent' | quality | 284 | 5 |
| 8453, 'trade', '' | quality | 90 | 5 |
| 8453, 'score', 'manual' | quality | 55 | 5 |
| 42220, 'epoch-fitness', 'aaveyielder' | quality | 53 | 5 |
| 42220, 'tycoon', 'agentaction' | quality | 38 | 5 |
| 8453, 'botcoin-skill', 'total_solves' | quantity | 25 | 5 |
| 8453, 'botcoin-skill', 'pass_rate' | quantity | 21 | 5 |
| 42220, 'spawn', 'celonova' | quality | 21 | 5 |

---

## 4. Kết quả benchmark — Pipeline Run 13

Cùng pipeline (BGE-SVM mandatory-escalation, thresh=0.8) chạy trên cả hai tập.

### 4.1 Bảng so sánh metrics

| Metric | Full pool (N=2,199*) | Stratified (N=1,322*) | Δ |
|---|---|---|---|
| **Macro F1** | 0.566 | 0.586 | +0.020 |
| **Weighted F1** | 0.879 | 0.865 | −0.014 |
| Quality F1 | 0.927 | 0.920 | −0.007 |
| Quality Recall | 0.938 | 0.936 | −0.002 |
| **Quantity F1** | 0.660 | 0.626 | −0.034 |
| **Quantity Recall** | 0.598 | 0.560 | −0.038 |
| Junk F1 | 0.111 | 0.211 | +0.100 |
| LLM calls (%) | 53.4% | 61.5% | +8.1pp |

*N_test < N_labelled vì pipeline chỉ test trên subset có đủ offchain context; một số records bị drop ở preprocessing step.

### 4.2 Diễn giải

**Macro F1 tăng nhẹ (+0.020):** Sau khi giảm concentration của các cluster quality lớn, tập stratified "harder" hơn về phần quality — nhưng đồng thời các cluster quality dễ không còn kéo Macro F1 xuống, nên Macro thực sự tăng một chút.

**Weighted F1 giảm nhẹ (−0.014):** Hợp lý vì weighted metric ưu ái class đa số (quality). Bộ stratified ít records quality hơn nên weighted F1 thấp hơn một chút.

**Quantity Recall giảm (0.598 → 0.560):** Điểm yếu chính vẫn còn nguyên sau stratification. Không phải do concentration — đây là limitation thật sự của pipeline khi gặp quantity tags.

**Junk F1 tăng (+0.100):** Junk chỉ có 3 records trong cả hai tập — con số này không có ý nghĩa thống kê, không nên diễn giải.

**LLM call rate tăng (+8.1pp):** Tập stratified có nhiều cluster nhỏ, ít lặp lại pattern → BGE SVM ít confident hơn → escalate LLM nhiều hơn. Đây là behaviour đúng.

---

## 5. Kết luận

1. **Kết quả full-pool không bị artifact của sampling concentration.** Macro F1 và Weighted F1 trên stratified subsample chỉ lệch ±2pp so với full pool.

2. **Điểm yếu quantity recall (0.56–0.60) là intrinsic**, không phải do Base chain hoặc một vài tag-pair lớn kéo kết quả. Nó xuất hiện ngay cả khi mỗi cluster chỉ được đại diện bằng ≤5 records.

3. **Imbalance quality:quantity (82%:17%) không thay đổi sau stratification**, xác nhận đây là phân bố thực tế của on-chain feedback ERC-8004, không phải artifact.

4. **Limitation còn lại:** Stratified check chỉ kiểm tra sampling robustness. Inter-annotator agreement (IAA) chưa được đo — toàn bộ nhãn do một annotator gán. Đây là future work.

---

## 6. Files liên quan

| File | Mô tả |
|---|---|
| `benchmarks/stratified_resample_gold.py` | Script tạo subsample |
| `data/labelled/pure_others_to_label.csv` | Full gold pool (N=2,206) |
| `data/labelled/pure_others_stratified_cap5.csv` | Stratified subsample (N=1,326, cap=5, seed=42) |
| `data/benchmark_results/pipeline_run13_20260622_235831.json` | Run 13 on full pool |
| `data/benchmark_results/pipeline_run13_20260624_160154.json` | Run 13 on stratified subsample |
