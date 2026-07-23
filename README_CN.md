# medstats

可嵌入后端的 Python 医学统计组件，面向肝病临床研究场景。

**不是 REST 服务。** 后端直接 `import`，传入 `pandas.DataFrame`，拿到 JSON-safe 字典，存库或返回前端。

## 安装

```bash
pip install git+https://github.com/bxjxis/medstats.git
```

## 功能一览

| 类别 | 函数 | 用途 |
|---|---|---|
| 生存分析 | `cox_regression` | Cox 比例风险模型，输出 HR、95% CI、p 值、C-index |
| 生存分析 | `kaplan_meier` | Kaplan-Meier 曲线坐标、中位生存时间、Log-rank p 值 |
| 回归 | `logistic_regression` | 二分类 Logistic 回归，输出 OR、95% CI、p 值、预测概率 |
| 模型评估 | `roc_analysis` | ROC 曲线、AUC 及 DeLong 法 CI、可选双模型比较 |
| 模型评估 | `calibration_analysis` | 校准曲线、Brier score、可选 Hosmer-Lemeshow 检验 |
| 模型评估 | `decision_curve_analysis` | 决策曲线分析（DCA），各阈值下净收益曲线 |
| 数据预处理 | `multiple_imputation` | 迭代插补（MICE 风格）处理数值型缺失值 |
| 因果推断 | `propensity_score_matching` | 1:1 最近邻倾向评分匹配，输出匹配对与 SMD 均衡表 |
| 因果推断 | `iptw_weights` | 稳定化 IPTW 权重，输出加权前后 SMD 均衡表 |

## 快速上手

```python
import pandas as pd
from medstats import cox_regression, kaplan_meier, roc_analysis

df = pd.read_csv("your_data.csv")

result = cox_regression(
    df,
    duration_col="time",       # 时间列
    event_col="event",         # 事件列（0/1）
    covariates=["age", "albumin", "stage"],
    categorical_cols=["stage"],  # 字符串/分类列 → 自动 dummy 编码
)

import json
json.dumps(result)  # 始终安全，无 numpy 类型，NaN → null
```

## 统一输出格式

每个函数都返回相同的外层结构：

```json
{
  "method": "cox_regression",
  "status": "ok",
  "params": {"duration_col": "time", "event_col": "event", "covariates": ["age", "albumin"]},
  "n_input": 300,
  "n_used": 285,
  "dropped_rows": 15,
  "warnings": [],
  "result": { ... }
}
```

| 字段 | 含义 |
|---|---|
| `status` | `ok` 或 `warning` |
| `n_input` | 输入总行数 |
| `n_used` | 去除缺失后实际使用行数 |
| `dropped_rows` | 因缺失被丢弃的行数 |
| `warnings` | 非致命的统计或数据质量提示 |
| `result` | 各方法特异性结果 |

后端可存储完整结构；前端从 `result` 读取数据，并展示 `warnings`。

## 行级输出与数据库关联

返回逐行结果的函数（`logistic_regression`、`multiple_imputation`、`iptw_weights`、`propensity_score_matching`）每行都带有 `id` 字段，方便后端 JOIN 回数据库记录。

通过 `id_col` 传入主键列：

```python
result = logistic_regression(df, "outcome", ["age", "albumin"], id_col="patient_id")
# result["result"]["predicted_probabilities"]
# → [{"id": "P001", "predicted_probability": 0.23}, ...]
```

不传 `id_col` 时默认使用 `df.index`。

## 分类协变量

字符串或分类列必须在 `categorical_cols` 中声明，否则抛出 `ValueError`。  
声明后自动执行 `pd.get_dummies(drop_first=True)` 编码。

```python
logistic_regression(df, "outcome", ["age", "stage"], categorical_cols=["stage"])
# 输出 term 名称包含 "stage_early"、"stage_advanced" 等
```

## 函数说明

### `cox_regression`

```python
cox_regression(df, duration_col, event_col, covariates, categorical_cols=None)
```

```json
"result": {
  "c_index": 0.74,
  "n_events": 88,
  "terms": [
    {"name": "age", "coef": 0.031, "hr": 1.031, "hr_ci_low": 1.01, "hr_ci_high": 1.05, "p": 0.002}
  ]
}
```

---

### `kaplan_meier`

```python
kaplan_meier(df, duration_col, event_col, group_col=None)
```

```json
"result": {
  "groups": [
    {"name": "early", "times": [0, 3, 6], "survival": [1.0, 0.97, 0.94], "median_survival": 42.0}
  ],
  "overall_logrank_p": 0.003
}
```

`times`（x 轴）/ `survival`（y 轴）→ 前端直接绘图。两组用两样本 Log-rank；三组及以上用多变量 Log-rank 整体 p 值。

---

### `logistic_regression`

```python
logistic_regression(df, outcome_col, covariates, add_intercept=True,
                    categorical_cols=None, id_col=None)
```

```json
"result": {
  "converged": true,
  "n_events": 72,
  "pseudo_r2": 0.18,
  "terms": [
    {"name": "age", "coef": 0.04, "or": 1.04, "or_ci_low": 1.01, "or_ci_high": 1.07, "p": 0.008}
  ],
  "predicted_probabilities": [
    {"id": "P001", "predicted_probability": 0.23}
  ]
}
```

---

### `roc_analysis`

```python
roc_analysis(df, label_col, score_col, compare_score_col=None)
```

`score_col` 必须在 `[0, 1]` 范围内。DeLong 方差使用 ddof=1（Sun & Xu 2014）。

```json
"result": {
  "auc": 0.81, "auc_ci_low": 0.76, "auc_ci_high": 0.86,
  "fpr": [...], "tpr": [...], "thresholds": [...],
  "compare": {
    "score_col": "model_b", "auc": 0.75,
    "fpr": [...], "tpr": [...],
    "delong_z": 1.82, "delong_p": 0.069
  }
}
```

`fpr`（x 轴）/ `tpr`（y 轴）→ 前端绘制 ROC 曲线。

---

### `calibration_analysis`

```python
calibration_analysis(df, label_col, score_col, n_bins=10, hosmer_lemeshow=True)
```

```json
"result": {
  "mean_predicted": [0.05, 0.15, ...],
  "fraction_observed": [0.04, 0.18, ...],
  "brier_score": 0.19,
  "hosmer_lemeshow": {"statistic": 8.4, "p": 0.39, "df": 8}
}
```

`mean_predicted`（x 轴）/ `fraction_observed`（y 轴）→ 校准图，完美校准为对角线。

---

### `decision_curve_analysis`

```python
decision_curve_analysis(df, label_col, score_cols, threshold_min=0.01,
                        threshold_max=0.99, n_thresholds=99)
```

```json
"result": {
  "thresholds": [0.01, 0.02, ...],
  "treat_all": [0.24, 0.23, ...],
  "treat_none": [0.0, 0.0, ...],
  "models": [{"name": "score_a", "net_benefit": [0.22, ...]}],
  "prevalence": 0.24
}
```

`thresholds` 为 x 轴，各曲线净收益为 y 轴。

---

### `multiple_imputation`

```python
multiple_imputation(df, target_cols, m=1, random_state=42, id_col=None)
```

使用 sklearn `IterativeImputer` 进行迭代插补。返回 `m` 份插补数据集。**不包含** Rubin 合并推断，如需合并请在调用层实现。

```json
"result": {
  "m": 1,
  "missing_rates": {"albumin": 0.05},
  "imputed_cols": ["albumin"],
  "datasets": [
    [{"id": 0, "albumin": 34.2}, {"id": 1, "albumin": 38.7}]
  ]
}
```

---

### `propensity_score_matching`

```python
propensity_score_matching(df, treatment_col, covariates, caliper=0.2,
                          replace=False, random_state=42, categorical_cols=None)
```

`caliper` 为 PS 标准差的倍数。`treated_id` / `control_id` 为原始 `df.index` 值，可直接 JOIN 数据库。

```json
"result": {
  "n_pairs": 142,
  "n_unmatched_treated": 8,
  "pairs": [{"treated_id": 5, "control_id": 23, "ps_diff": 0.003}],
  "balance_before": [{"covariate": "age", "mean_treated": 56.1, "mean_control": 54.2, "smd": 0.19}],
  "balance_after":  [{"covariate": "age", "mean_treated": 55.8, "mean_control": 55.4, "smd": 0.04}],
  "matched_data": {"id": [5, 23, ...], "treatment": [1, 0, ...]}
}
```

---

### `iptw_weights`

```python
iptw_weights(df, treatment_col, covariates, clip_quantiles=(0.01, 0.99),
             random_state=42, categorical_cols=None, id_col=None)
```

```json
"result": {
  "rows": [
    {"id": 0, "propensity_score": 0.43, "weight": 1.12}
  ],
  "balance_before":         [{"covariate": "age", "smd": 0.19}],
  "balance_after_weighted": [{"covariate": "age", "smd_weighted": 0.02}]
}
```

---

## 接入说明

**后端**
- 调用前验证用户选择的列名是否存在。
- 建议存储完整 envelope，便于审计和复现。
- 将 `warnings` 写入日志或传给下游消费者。
- 使用 `id_col` 或依赖 `df.index` 将逐行结果 JOIN 回数据库记录。

**前端**
- 后端不返回图片，曲线以坐标数组形式传输，前端自行绘图。
- 在结果面板附近展示 `warnings`。
- `status == "warning"` 可用于显示非阻塞的数据质量提示。

## 运行测试

```bash
git clone https://github.com/bxjxis/medstats.git
cd medstats
pip install -e ".[dev]"
pytest -q
```

预期结果：**38 passed**
