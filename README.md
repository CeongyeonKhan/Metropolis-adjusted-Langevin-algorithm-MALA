# 能量—梯度误差约束自适应混合 MALA

本工程依据《能量梯度误差约束自适应混合 MALA 实验方案》实现，默认完成解析合成实验 P0—P7，并使用 PyTorch 进行向量化采样。正式采样中的接受能量固定为解析目标能量 \(E^*\)；冻结梯度集成只用于构造候选点。该边界保证梅特罗波利斯—黑斯廷斯（Metropolis–Hastings, MH）校正消除的是给定目标下的离散化偏差与提议梯度误差，而不是把学习目标误差误解释为采样误差。

项目同时记录本地 JSONL/CSV 结果和 Weights & Biases（W&B）运行。W&B 支持在线、离线和禁用三种模式；默认配置采用离线模式，不需要登录。

## 1. 已实现内容

- 解析单高斯、对称/非对称双模态、环形、网格及高维混合高斯；
- 解析能量、责任度、梯度和海森矩阵；
- 冻结随机傅里叶特征梯度集成，以及尺度、正交和边界增强压力误差；
- 40% 目标样本、40% 模态桥、20% 宽高斯组成的独立校准点；
- 盆地软责任度加权中位数与绝对中位差标准化；
- 拟合、覆盖校准、测试三划分的梯度误差上界；
- 能量、曲率、梯度误差、漂移距离及最大步长约束；
- 状态相关局部 MALA，候选点处完整重算反向提议参数；
- 具有显式密度的高斯—多元学生 \(t\) 独立全局提议；
- B0、B1、B2、A1、A2-O、A2-P、A3、A4、A4-NC；
- 秩归一化折叠分裂 \(\widehat R\)、主体/尾部有效样本量（Effective Sample Size, ESS）、原始空间弗雷歇距离、模态占用 JS 散度；
- 固定监测网格上的首次跨模态右删失、逐链成本 Kaplan–Meier 曲线、限制平均跨模态成本及转移矩阵；
- 能量、梯度、曲率和全局密度计算的等效成本；
- 机制热图、轨迹、适配散点图、约束激活图、诊断曲线、存活曲线和转移矩阵。

学习能量网络和扩散初始化属于方案中的 P8 补充实验，首版没有并入解析主实验，避免同时改变接受目标、梯度质量和初始化制度。

## 2. 安装

建议使用 Python 3.10 或更高版本。以下命令均应在解压后的项目根目录执行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

若只运行实验而不执行测试，也可使用：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

安装完成后可用 `ege-mala --help` 或 `python -m ege_ah_mala --help` 检查命令行入口；两者等价。

## 3. 首次运行

先运行 CPU 烟雾测试。该配置包含 8 条链、120 步及 4 个方法，并在本地生成 W&B 离线运行：

```bash
python -m ege_ah_mala.cli run --config configs/smoke.yaml
```

在线记录：

```bash
wandb login
python -m ege_ah_mala.cli run \
  --config configs/p1_bimodal.yaml \
  --wandb-mode online
```

完全关闭 W&B，但仍保留本地 JSONL 和 CSV：

```bash
python -m ege_ah_mala.cli run \
  --config configs/smoke.yaml \
  --wandb-mode disabled
```

同步离线运行时，使用终端中 W&B 给出的离线目录执行 `wandb sync`。程序不会在代码中调用 `wandb.login()`，认证信息只由 W&B 自身读取。

每个方法对应一条独立 W&B 运行，运行名称为 `<实验名>-<方法>-seed<种子>`，相同实验内的方法通过组名关联。`wandb.mode=online` 实时上传指标，`offline` 仅写本地离线记录，`disabled` 完全不初始化 W&B。若初始化失败且 `wandb.strict=false`，程序写入 `wandb_error.txt` 后继续保留本地指标；要求 W&B 失败即终止时应设 `wandb.strict=true`。

纯 CPU 长任务若不需要 W&B 的主机系统指标，可按需关闭后台统计采集，减少额外轮询开销：

```bash
WANDB_X_DISABLE_STATS=true python -m ege_ah_mala.cli run \
  --config configs/p5_full.yaml \
  --wandb-mode online
```

该环境变量对应 W&B 的内部设置 `Settings.x_disable_stats=true`，只关闭 CPU、内存等系统指标，不影响本项目显式记录的采样指标。该 `x_` 设置属于 W&B 内部接口，升级 W&B 后应重新核验；不需要任何 W&B 记录时应直接使用 `--wandb-mode disabled`。

## 4. 配置覆盖

命令行可重复使用 `--set`，无需复制配置文件：

```bash
python -m ege_ah_mala.cli run \
  --config configs/p5_full.yaml \
  --methods B1,B2,A4 \
  --steps 2000 \
  --chains 32 \
  --set error.relative_rmse=0.1 \
  --set adaptation.gamma=0.0 \
  --set output.experiment_name=gamma0_check
```

主要配置如下。

| 配置节 | 作用 |
|---|---|
| `target` | 目标类型、维数、模态数、间隔、条件数和权重 |
| `whitening` | 固定仿射白化开关、尺度来源和正定扰动量 |
| `reference` | 独立预热混合拟合、样本数、EM 迭代数及协方差正则化 |
| `error` | 冻结梯度集成、随机特征数、误差类型与相对均方根误差 |
| `calibration` | 校准点数、三划分比例、覆盖率和宽高斯尺度 |
| `adaptation` | \(h_0,h_{\max},\alpha_E,\tau_{\max},\gamma,c_L,\varepsilon_{\rm prop}\) 等 |
| `global_proposal` | 固定路由概率、两类尺度、学生 \(t\) 权重和自由度 |
| `sampler` | 方法、链数、预算、初始化、诊断与轨迹间隔 |
| `metrics` | 原始空间弗雷歇距离基准重复数、模态确认规则 |
| `cost` | 能量、曲率、全局密度及扩散调用的成本权重 |
| `wandb` | online/offline/disabled、项目、分组、标签与制品 |
| `output` | 输出根目录、实验名、历史保存和绘图分辨率 |

## 5. 方法定义

| 编号 | 固定/自适应 | 误差约束 | 自适应噪声 | 全局提议 | MH 校正 |
|---|---|---|---|---|---|
| B0 | 固定 | 无 | 无 | 无 | 无 |
| B1 | 固定 | 无 | 无 | 无 | 有 |
| B2 | 固定 | 无 | 无 | 有 | 有 |
| A1 | 能量、曲率、漂移 | 无 | 无 | 无 | 有 |
| A2-O | A1 | 真实误差 | 无 | 无 | 有 |
| A2-P | A1 | 代理上界 | 无 | 无 | 有 |
| A3 | A2-P | 代理上界 | 有 | 无 | 有 |
| A4 | A3 | 代理上界 | 有 | 有 | 有 |
| A4-NC | A4 | 代理上界 | 有 | 有 | 无 |

A2-O 是不可部署的预言机性能上界。A4-NC 同时无条件接受局部和全局候选，因此仅用于稳态偏差消融，不能作为正确采样器。

## 6. 必须保留的正确性约束

通用局部提议实现为

\[
\mu_x=x-h_x\tau_x^\gamma\widehat g_{\rm prop}(x),
\qquad
v_x=2h_x\tau_x .
\]

因此 \(\gamma=0\) 时漂移项不会错误地乘以 \(\tau\)。反向密度在候选点完整重算 \(z_E,z_g,\delta_U,\widehat L,h,\tau,\widehat g_{\rm prop}\)。正式采样阶段不更新模型、盆地统计、校准系数、曲率定义、全局提议或 \(h_0\)。

局部—全局路由概率 \(\rho\) 在整个正式阶段固定。不得依据当前位置、能量或“局部步长过小”改走全局分支；若将来引入状态相关路由，必须在接受率中加入正反向分支选择概率。

当 \(\gamma=1\) 时，局部核只依赖有效步长 \(s=h\tau\)，所以 \(h\) 与 \(\tau\) 在转移核层面不可独立辨识。工程同时记录扩散尺度 `sampler/effective_step_*` 与漂移尺度 `sampler/drift_scale_*`，并提供 `configs/p4_noise_only_gamma0.yaml` 作为能够检验随机项独立作用的主要对照。

多元学生 \(t\) 的 `student_scale * covariance` 在本项目中定义为尺度矩阵，而不是协方差矩阵；采样和密度使用完全相同的约定。

## 7. 误差上界校准

项目先在拟合集上求非负系数

\[
\delta^{*2}\approx a_\delta u_\delta^2+b_\delta ,
\]

再在独立覆盖校准集上使用有限样本比值分数

\[
s_i=\frac{\delta_i^{*2}}
{\max(a_\delta u_{\delta,i}^2+b_\delta,\varepsilon)}
\]

确定经验分位数 \(c_{0.95}\)，最终使用

\[
\delta_U^2=c_{0.95}
\max(a_\delta u_\delta^2+b_\delta,\varepsilon).
\]

剩余独立测试集用于报告总体、模态内部、模态边界和高能区域覆盖率。这里得到的是相对于校准分布的边际经验覆盖，不是所有条件子区域的严格概率保证。

## 8. W&B 记录

所有正式采样曲线使用每条链等效梯度成本 `axis/ceq_per_chain` 作为横坐标。主要字段包括：

- `sampler/h_*`、`sampler/tau_*`、`sampler/effective_step_*`、`sampler/drift_scale_*`；
- `constraints/fixed_base`、`energy`、`curvature`、`gradient_error`、`drift`、`hmax`、`numeric_floor`；
- `error/delta_true_*`、`error/delta_upper_*`、`coverage`；
- `accept/local_*`、`global_*`、`overall_*`；
- `diagnostics/rhat_*`、`ess_bulk_min`、`ess_tail_min`；
- `distribution/fid_raw`、`distribution/js`、`distribution/min_mode_ratio`；
- `crossing/censored_fraction`、`rmst_cost`、`switches_per_1000_cost`；
- `cost/n_energy`、`n_gradient`、`n_hessian`、`n_global_density`；
- `runtime/sampling_s`、`runtime/metrics_s`、`runtime/peak_cuda_mb`；
- `health/nonfinite_energy`、`nonfinite_gradient`、`auto_reject`。

若某个模态指示量在各链内为常数但链间不同，内部 \(\widehat R\) 保留为无穷大；W&B 另记录 `diagnostics/rhat_has_infinite=1`，不会用显示截断值进行收敛判定。

默认的拟合参考分布不读取解析混合参数：程序使用命名随机流 `reference_fit` 独立抽取预热样本，再通过期望最大化（Expectation–Maximization, EM）拟合高斯混合。该参考分布与误差尺度点、盆地标准化点及误差校准点使用不同随机流，并在正式采样前冻结；参数和拟合元数据写入 `reference_mixture.json`。`reference.source=analytic` 仅用于预言机消融。

## 9. 输出目录

每次实验写入：

```text
outputs/<experiment_name>/seed_<seed>/
├── resolved_config.yaml
├── environment.json
├── target.json
├── reference_mixture.json
├── error_field.json
├── normalizer.json
├── calibration.json
├── frozen_adaptation.pt
├── initial_states.pt
├── fid_reference_band.json
├── method_capabilities.json
├── method_summary.csv
├── method_summary.json
├── run_manifest.json
├── <method>/
│   ├── resolved_config.yaml
│   ├── metrics.jsonl
│   ├── metrics.csv
│   ├── summary.json
│   ├── samples.pt
│   ├── plots/
│   └── wandb/
└── a4_vs_a4_nc.png
```

`samples.pt` 默认同时保存白化工作坐标与原始坐标下的终态和稀疏轨迹，并保存诊断历史、对应步号/成本、转移矩阵和存活曲线。大规模实验可设 `output.save_history=false`。

输出路径只由 `output.root`、`output.experiment_name` 和 `seed` 决定，程序不会自动生成唯一目录，也不把已有目录视为可恢复的检查点。重复使用同一路径会覆盖同名配置、指标和样本文件，同时可能保留本次未生成的旧图或旧 W&B 文件。因此，每次正式运行应使用新的实验名、种子或输出根目录；仅在明确需要覆盖一次失败试跑时复用路径，并先人工核对或另行归档原结果。例如：

```bash
python -m ege_ah_mala.cli run \
  --config configs/p5_full.yaml \
  --set output.experiment_name=p5_full_20260829_r1
```

## 10. 实验顺序

建议依次运行：

1. `configs/p0_single_gaussian.yaml`；
2. `configs/p1_bimodal.yaml`；
3. `configs/p2_oracle_error.yaml`；
4. `configs/p3_proxy_error.yaml`；
5. `configs/p4_adaptive_noise.yaml` 与 `p4_noise_only_gamma0.yaml`；
6. `configs/p5_full.yaml`；
7. `configs/p6_uncorrected_bias.yaml`；
8. `configs/p7_ring.yaml`、`p7_grid.yaml`、`p7_highdim.yaml`。

正式配置的预算明显高于烟雾测试，CPU 运行可能持续数小时；先用命令行覆盖缩短步数，确认设备、输出目录和 W&B 模式后再提交长任务。

## 11. 测试

```bash
pytest
ruff check src tests scripts
```

测试覆盖解析梯度/海森矩阵及海森矩阵—向量积、通用 \(\gamma\) 详细平衡恒等式、步长上界、全局混合密度、提议复现性、误差校准、\(\widehat R\)/ESS、逐链成本右删失存活曲线和端到端命令行运行。

设备成本微基准与多种子汇总分别使用：

```bash
python scripts/benchmark_costs.py --config configs/p5_full.yaml --output cost_weights.json
python scripts/aggregate_results.py outputs/p5_full --output p5_all_seeds.csv
```

W&B 参数扫描配置为 `configs/wandb_sweep.yaml`。先在独立验证目标和验证种子上执行：

```bash
wandb sweep configs/wandb_sweep.yaml
wandb agent <返回的扫描标识>
```

扫描包装器固定只运行 A4，并以归一化 JS 曲线积分加覆盖率/非有限值惩罚作为 `sweep/objective`。正式测试种子不应参与参数选择。

## 12. 当前阶段边界

- 主实验在解析工作坐标中运行，接受目标始终为 \(E^*\)。
- 参考配置默认设置 `whitening.enabled=true`，在预热阶段按混合分量池化协方差构造固定仿射变换，并同时保存原坐标与工作坐标参数；正式采样中不再更新该变换。`FID_raw`、均值误差和协方差误差均在逆变换后的原始坐标计算。需要研究不白化压力时可显式设为 `false`。
- 默认从独立命名随机流产生的预热目标样本拟合参考混合，不直接复用解析目标分量参数；参考责任度与全局提议均在正式采样前冻结。可用 `reference.source=analytic` 运行预言机参考消融，但该结果不得解释为可部署性能。
- 低维配置默认对冻结梯度场雅可比执行精确最大奇异值计算；高维配置使用固定探针、固定轮数的海森矩阵—向量积幂迭代，并按实际乘积次数计费。二者均乘 1.5 安全系数。该量是逐点局部稳定代理，不是包含 \(h(x)\) 与 \(\tau(x)\) 空间导数后的严格全局上界。
- 成本权重是配置中的工程默认值。论文级比较前应在目标设备上微基准测量后固定，并将预热拟合的端到端时间一并报告。
