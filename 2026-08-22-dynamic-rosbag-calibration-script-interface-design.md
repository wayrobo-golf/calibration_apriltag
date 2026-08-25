# 动态标定软件接口及 Web/APP 集成需求说明书

> 文档编号：VUP-DYN-CAL-SRS-001
> 版本：V2.1
> 日期：2026-08-25
> 状态：正式交付稿
> 适用对象：设备端 APP、Web、测试、现场 FAE
> 适用范围：单设备、本地 rosbag、离线动态标定；一期不负责结果安装

## 修订记录

| 版本 | 日期 | 修订内容 |
|---|---|---|
| V1.0 | 2026-08-22 | 初版 |
| V1.1 | 2026-08-22 | 明确 `robot_map/base_pnt` 的 UTM 基准点语义 |
| V1.2 | 2026-08-24 | 补充设备实测基准点及坐标约束 |
| V1.3 | 2026-08-24 | 增加码牌检测、标定和安装状态要求 |
| V2.0 | 2026-08-25 | 明确独立算法交付物和单配置接口 |
| V2.1 | 2026-08-25 | 将建库和配置合并从软件需求中移除，按已交付接口重新整理 Web/APP 需求 |

## 1. 文档目的

本文向 APP/Web 开发人员定义：

1. 已交付动态标定软件如何调用；
2. 一次标定任务需要提供什么输入；
3. 当前算法实际产生什么输出；
4. 数据录制脚本如何使用；
5. APP/Web 需要实现的任务管理、状态指示、地图和 UTM 坐标显示要求；
6. 一期禁止自动安装标定结果的边界。

动态标定库的建立、旧算法迁移及三份配置合并已经由算法侧完成，不是 APP/Web 的开发需求。APP/Web 只依赖本文件规定的外部接口，不依赖旧日期脚本和旧三配置组合。

## 2. 交付基线

算法侧同步交付以下内容：

| 交付物 | 仓库位置 | 用途 |
|---|---|---|
| 独立 Python 包 | `vision_unload_platform/tools/dynamic_rosbag_calibration/` | 标定算法、构建文件、测试和 CLI |
| 唯一任务配置 | `vision_unload_platform/tools/dynamic_rosbag_calibration/config/calibration_job.yaml` | 当前 20260821 数据的可运行配置和后续配置模板 |
| 包使用说明 | `vision_unload_platform/tools/dynamic_rosbag_calibration/README.md` | 安装、调用和输出说明 |
| 本需求说明书 | 本文 | APP/Web 集成和验收依据 |
| 数据录制脚本 | `/home/share_data/gjf/unloading_using_apriltag_and_pnp/20260821/2_rosbag/rosbag_toggle.sh` | 启停本地 rosbag 录制 |

交付包具有独立构建元数据，运行时不导入旧 `fae_calibration` 包。旧入口 `run_20260820_same_day_dynamic_calibration.py` 仅作为算法回归基线，APP/Web 不得调用。

## 3. 范围和职责边界

### 3.1 一期范围

- 单台设备使用独立页面；
- 本地录制一组 rosbag；
- 从一组 rosbag 中构造多个两-bag 标定候选；
- 对候选执行同日留出验证、稳定性比较和排序；
- 质量或审计门限由任务配置提供；
- PASS 和 FAIL 均保留已成功求解的数值候选；
- 在设备地图上显示机器人轨迹和最终固定码牌位置；
- 显示 WGS-84 / UTM 投影带、Easting、Northing 和高度；
- 分开显示码牌检测状态、标定状态和安装状态。

### 3.2 一期不包含

- 不修改、覆盖或切换设备正式运行配置；
- 不把候选自动写入 `tf_static_golfC_param.yaml`；
- 不安装标定结果；
- 不要求算法封装成指定的 ROS 2 Action、Service 或 HTTP 服务；
- 不负责 rosbag 上传、云存储和跨设备调度；
- 标定完成后不根据机器人实时位置持续更新码牌位置。

### 3.3 责任划分

| 模块 | 职责 |
|---|---|
| 动态标定库 | 校验单配置、读取 rosbag、求解候选、验证、排序并写出算法结果 |
| APP | 调用录制脚本、生成任务配置、启动/停止所选封装、管理本机任务和文件 |
| Web | 每设备独立页面、进度与告警、状态卡片、地图和结果下载 |
| APP/Web 集成层 | 将算法输出映射成产品业务状态，完成 UTM 展示及历史结果管理 |
| 人工审核流程 | 决定候选是否进入一期之外的发布或安装流程 |

APP/Web 可自行选择普通子进程、ROS 节点、Action、Service 或其他封装；选择不改变算法输入、输出及安全边界。

## 4. 数据录制脚本

### 4.1 调用方式

脚本采用 toggle 语义，没有 `start`、`stop`、`status` 子命令：

```bash
# 第一次执行：开始录制
bash /home/share_data/gjf/unloading_using_apriltag_and_pnp/20260821/2_rosbag/rosbag_toggle.sh

# 第二次执行：停止当前录制
bash /home/share_data/gjf/unloading_using_apriltag_and_pnp/20260821/2_rosbag/rosbag_toggle.sh
```

传入 `start` 或 `stop` 参数不会改变脚本语义。APP 在调用前必须检查当前 PID/进程状态，避免一次重复调用把“开始”变成“停止”。

### 4.2 录制输出

脚本所在目录下生成：

```text
bag_YYYYMMDD_HHMMSS/       rosbag目录
bag_YYYYMMDD_HHMMSS.log    ros2 bag record日志
rosbag_record.pid          仅录制期间存在
```

APP 应识别标准输出前缀：

| 前缀 | 含义 |
|---|---|
| `[START]` | 录制已启动，并给出 PID、bag 和日志路径 |
| `[STOP]` | 正在停止或已经停止 |
| `[WARN]` | 检测到残留或异常状态 |
| `[ERROR]` | 启停或文件操作失败 |

### 4.3 录制话题

当前脚本录制：

```text
/tf
/tf_static
/camera/left/image
/camera/left/intrinsic_matrix
/novatel/oem7/inspvax
/novatel/oem7/ins_odom
/novatel/oem7/robot_map/base_pnt
/localization/m2_basic_msg
/livox/imu
/livox/pointcloud
/novatel/oem7/robot_odom
/novatel/oem7/imu/data_raw
/novatel/oem7/imu/data
```

标定数学链路至少需要图像、`ins_odom`、INSPVAX 和 IMU。要求显示绝对 UTM 码牌坐标时，还必须实际录到 `/novatel/oem7/robot_map/base_pnt`，不能只确认在线话题存在。

### 4.4 每包录制流程

1. 确认相机、INS、IMU、`base_pnt` 话题正常；
2. 启动录制；
3. 等待订阅建立；
4. 机器人完成一趟独立动态轨迹；
5. 轨迹内保持 Tag 0、Tag 1 尽量同时可见，并覆盖不同距离和横向视角；
6. 停车后停止录制；
7. 等待 `metadata.yaml` 和 db3/mcap 文件写完；
8. 执行 bag 完整性检查后再录下一包。

### 4.5 停止后的验收

仅脚本退出码为 0 不代表数据合格。APP 至少检查：

- bag 目录、`metadata.yaml` 和存储文件存在；
- 必需话题存在且消息数大于 0；
- 图像、odom、INSPVAX、IMU 时间范围有效；
- 需要 UTM 时，`base_pnt` 至少有一条有效消息；
- 存储文件可以读取；
- PID 文件已经清理且录制进程退出；
- 本包路径与其他包不重复。

当前脚本为单 PID 文件设计，同一目录禁止并行录制两个 bag。

## 5. 标定调用接口

### 5.1 CLI

安装包后，APP/Web 若采用子进程方式，只调用：

```bash
dynamic-rosbag-calibration --config /absolute/path/calibration_job.yaml
```

生产入口只有一个业务参数 `--config`。旧参数 `--data-root`、`--manifest`、`--base-experiment-config` 等不再对外使用。

部署时必须按包元数据安装依赖，不能只把源码目录加入 `PYTHONPATH`。缺少 `pyapriltags` 等运行依赖时，入口必须以非零状态失败，不得生成可被误认为有效的空标定结果。

### 5.2 Python 接口

上层 Python 进程可以先校验、再同步执行：

```python
from dynamic_rosbag_calibration import load_job_config, run_calibration

job = load_job_config("/absolute/path/calibration_job.yaml")
exit_code = run_calibration(str(job.source_path))
```

当前稳定公共符号为：

| 符号 | 说明 |
|---|---|
| `load_job_config(path)` | 严格解析唯一 YAML，校验重复字段、角色、路径和基础类型 |
| `CalibrationJobConfig` | 只读任务配置对象 |
| `RosbagInput` | 单个 bag 的 ID、路径、角色和可选排除原因 |
| `run_calibration(path)` | 同步执行任务，返回进程式整数状态 |

任务业务 PASS/FAIL 不得只按整数返回值判断，必须读取结果目录中的汇总和候选质量字段。

### 5.3 标准输出

算法使用一行一个 JSON 对象输出进度，例如：

```json
{"stage":"same_day_bag_read_start","date":"20260821","bag_id":"104504","bag_index":1,"bag_count":15}
```

APP/Web 应逐行解析：

- 可解析 JSON 行用于更新进度；
- 第三方库警告写入日志，不得导致整个进度流失效；
- 长时间无阶段更新可告警，但不得用固定总时长直接判断失败；
- 最终业务状态以结果文件为准。

## 6. 唯一配置输入

### 6.1 配置位置和原则

交付样例：

```text
vision_unload_platform/tools/dynamic_rosbag_calibration/config/calibration_job.yaml
```

一次任务只提交这一份 YAML。机器内参、外参和 Tag 几何初值仍作为机器资产目录中的文件存在，由唯一配置引用，不属于额外任务配置。

配置顶层字段为：

```text
schema_version
job
runtime
machine
rosbags
topics
detection
interpolation
structural_gate
sampling
coverage
calibration
```

字段语义以交付样例和配置加载器为准。解析器拒绝重复 YAML key、未知公共字段、重复 bag ID、缺失路径和缺失 `metadata.yaml`。

### 6.2 rosbag 角色

| 角色 | 是否参与拟合 | 是否参与普通验证 | 说明 |
|---|---:|---:|---|
| `calibration` | 是 | 对未参与本候选的 bag 是 | 至少 2 包，用于构造两-bag候选 |
| `validation` | 否 | 是 | 至少 1 包，独立于拟合集合 |
| `displacement_evaluation` | 否 | 单独统计 | 用于特殊位移评估，不参与候选生成和排序 |
| `excluded` | 否 | 否 | 必须给出原因，保留审计记录 |

每项均包含稳定 `id` 和本地 rosbag 目录 `path`。同一 ID 不能出现在两个角色中。

### 6.3 路径和并发要求

- 输入必须是本机可读的 rosbag 目录；
- 输出目录由 `runtime.output_dir` 指定；
- 每个任务使用独立输出目录；
- 不允许两个任务并发写同一输出目录；
- 输出目录不得位于设备正式运行配置目录；
- 算法不得删除或修改输入 bag；
- 相对路径按配置文件所在目录解析，APP 推荐写绝对数据路径。

### 6.4 可调门限

算法侧门限均由唯一配置提供，主要包括：

- AprilTag 检测和几何一致性门限；
- pose 插值时间门限；
- 动态结构状态门限；
- 抽样增量、每包帧数和总帧数门限；
- 距离区间与 bearing 覆盖门限；
- G2 几何质量门限；
- B2 求解器、可观性和多起点一致性门限；
- `calibration.quality_gate.position_limit_m`；
- `calibration.quality_gate.yaw_limit_deg`；
- 是否要求全部验证项通过。

门限在一次任务开始后不得由 UI 原地修改。调整门限必须生成新的 `job.id` 和新输出目录，并保留对应配置文件，以便审计和复现。

`allow_failed_candidate_for_diagnostics=true` 只允许质量失败后继续产生诊断候选，不得把失败候选标为可安装。

## 7. 当前算法语义

“一组 rosbag 联合标定”的当前含义为：

```text
输入一组 bag
  -> 逐 bag 解码和证据提取
  -> 枚举满足覆盖要求的两-bag组合
  -> 确定性选择有限数量组合
  -> 每个组合独立执行 G2 + 冻结几何复检 + B2-RXY
  -> 使用未参与该候选拟合的同日 bag 做留出验证
  -> 比较候选稳定性和验证误差
  -> 选择代表候选
```

它不是把全部 bag 一次性放入一个 B2 优化器。

坐标变换统一使用：

```text
p_A = T_A_B * p_B
```

关键候选字段：

| 字段 | 实际语义 |
|---|---|
| `T_tag0_tag1_calibrated` | `p_tag0 = T_tag0_tag1 * p_tag1` |
| `T_ins_camera_calibrated` | `p_ins = T_ins_camera * p_camera` |
| `T_ins_map_tag0_calibrated` | 历史字段名；实际为 `p_world = T_world_tag0 * p_tag0` |

Web/APP 不得仅根据字段名猜测方向。

## 8. 当前算法输出

### 8.1 目录

当前实际输出的主要文件为：

```text
<output_dir>/
  summary.json
  REPORT.md
  <YYYYMMDD>/
    summary.json
    REPORT.md
    bag_8m_inventory.csv
    bag_read_failures.json
    combination_plan.json
    candidate_parameters.csv
    candidate_ranking.csv
    selected_candidate.json
    pairwise_transform_differences.csv
    transform_stability_summary.csv
    SELECTED_INS_VISUAL_TRAJECTORY.html
    candidates/
      <candidate_id>/
        combination_input.json
        recovered_candidate.yaml
        recovery_audit.json
        experiment_metadata.json
        failure.json
    validation_same_day/
      per_bag_metrics.csv
      candidate_retention.csv
      validation_failures.json
      validation_exclusions.json
      interactive_bags/index.html
```

`failure.json` 只在对应候选执行异常时出现。HTML/CSV/JSON 均为运行产物，不应提交到代码仓库。

### 8.2 APP/Web 最少读取集合

1. 根 `summary.json`；
2. 日期级 `summary.json`；
3. `selected_candidate.json`；
4. 选中候选目录的 `recovered_candidate.yaml`；
5. 同目录的 `recovery_audit.json`；
6. `candidate_ranking.csv`；
7. `validation_failures.json` 和 `validation_exclusions.json`；
8. `REPORT.md` 或 HTML 轨迹入口。

### 8.3 候选保留规则

- `recovered_candidate.yaml` 是成功求解得到的数值候选；
- `quality_gate_pass=false` 时仍保留该文件及审计；
- G2 失败但允许诊断继续时，后续收敛不得抹去 G2 失败事实；
- 输入解码失败、资产缺失或求解前异常时可能没有数值候选；
- 不得为了满足“失败也有候选”而伪造矩阵；
- 一期所有候选均为 `NOT_INSTALLED`。

## 9. APP/Web 业务状态映射

### 9.1 任务状态

产品层统一显示：

| 状态 | 中文 | 候选要求 |
|---|---|---|
| `NOT_STARTED` | 尚未开始 | 无 |
| `RUNNING` | 标定中 | 无 |
| `PASS` | 标定通过 | 至少一个成功候选，选中候选质量和要求的验证通过 |
| `FAIL` | 标定未通过 | 至少一个数值候选，但质量或验证未通过 |
| `ERROR` | 执行异常 | 不保证有候选 |
| `CANCELED` | 已取消 | 已生成内容仅用于审计 |

算法退出码为 0 仅代表流程正常结束，不等价于业务 PASS。业务层至少结合根/日期摘要、`selected_candidate_id`、候选 `quality_gate_pass` 和验证失败清单进行映射。

### 9.2 历史合格结果状态

设备已有结果与最近一次任务必须分开：

| 状态 | 中文 | 说明 |
|---|---|---|
| `NONE` | 未标定 | 没有与当前设备身份匹配的合格结果 |
| `AVAILABLE` | 已标定，存在合格结果 | 有身份、文件校验和质量门均有效的历史 PASS |
| `STALE` | 已有结果但已过期 | 超过 APP/Web 配置的新鲜度期限 |
| `INCOMPATIBLE` | 已有结果但不匹配 | 设备、相机、码牌、配置或软件身份变化 |

最近任务 FAIL、ERROR 或 CANCELED 时，不得自动删除仍有效的历史 `AVAILABLE` 结果。页面应允许同时显示“已有历史合格结果；最近一次重标定失败”。

### 9.3 安装状态

安装状态必须独立显示。虽然产品模型可保留 `NOT_INSTALLED / INSTALLED / UNKNOWN`，但一期固定为：

```text
installation_state: NOT_INSTALLED
```

APP/Web 不提供安装按钮，不复制候选到运行目录，不修改运行配置。

## 10. 每设备页面显示要求

每台设备拥有独立 Web 页面。首屏必须同时显示以下三类状态，不能只放在日志详情中：

1. 码牌检测状态；
2. 码牌标定状态；
3. 安装状态。

每个状态使用文字、图标和颜色共同表达，不能只依赖颜色。

### 10.1 码牌检测状态

总体状态：

| 状态 | 中文显示 |
|---|---|
| `UNKNOWN` | 未检查 |
| `CHECKING` | 检查中 |
| `NONE_DETECTED` | 当前数据未检测到码牌 |
| `PARTIAL_DETECTED` | 部分码牌已检测 |
| `ALL_REQUIRED_DETECTED` | 所需码牌已检测 |

Tag 0、Tag 1 分别显示 `UNKNOWN / NOT_DETECTED / DETECTED`，并显示有效帧数、最后检测时间和主要失败原因。

状态必须携带证据来源和时间：

```text
evidence_source: LIVE_CAMERA / ROSBAG_REPLAY / LAST_TASK_RESULT
freshness: LIVE / RECENT / HISTORICAL / STALE
source_bag_ids
source_stamp
updated_at
```

历史 rosbag 只能证明历史数据中检测到码牌，不能直接证明“当前场地实时存在”。单次未检测也不能写成“场地不存在码牌”，应写“当前数据未检测到码牌”。

### 10.2 标定状态卡片

至少显示：

- 最近任务状态、任务 ID 和完成时间；
- 当前或选中候选 ID；
- PASS/FAIL 和主要原因；
- 标定 bag 数、验证 bag 数和排除 bag 数；
- 候选及审计文件下载入口；
- 历史合格结果状态；
- “一期未安装”固定提示。

检测到码牌、完成求解、质量通过和已经安装是四个不同事实，不得合并成一个绿色图标。

### 10.3 地图显示

标定期间页面至少显示：

- 当前处理 bag 和阶段；
- 机器人历史轨迹；
- 可选的临时码牌观测点；
- 被排除观测及原因；
- 质量告警。

标定完成后：

- 地图显示选中候选的最终码牌位置；
- 该位置切换为 `FINAL_FROZEN`；
- 后续机器人移动不得更新该码牌坐标；
- 新一次标定任务完成前，继续显示当前有效的冻结结果；
- 新任务失败不得自动覆盖旧的有效冻结结果；
- FAIL 候选可预览，但必须使用红色边框或明确“不可部署”标识。

建议图例：

| 对象 | 样式 |
|---|---|
| 机器人轨迹 | 蓝色细线 |
| 当前机器人 | 蓝色带朝向三角形 |
| 临时有效观测 | 半透明橙色点 |
| 排除观测 | 灰色或红色叉号 |
| 最终 PASS 码牌 | 绿色实心标记 |
| 最终 FAIL 候选 | 红色边框标记 |
| 验证轨迹 | 紫色虚线 |

## 11. WGS-84 / UTM 坐标显示

### 11.1 数据来源

| 数据 | 来源 | 语义 |
|---|---|---|
| map 原点绝对坐标 | `/novatel/oem7/robot_map/base_pnt` | `geometry_msgs/msg/Vector3`，x/y/z 为 UTM Easting、Northing、高度 |
| 机器人 map 局部位姿 | `/novatel/oem7/ins_odom` | 相对 map 原点的局部坐标 |
| 经纬高和质量 | `/novatel/oem7/inspvax` | WGS-84 经纬高、UTM 带推导和一致性检查 |
| 码牌 map 位姿 | 选中候选 `T_ins_map_tag0_calibrated` | 码牌在 `ins_map/world` 下的冻结位姿 |

设备实测 `base_pnt`：

```yaml
x: 676668.9018173097
y: 3120669.376408973
z: 49.0
```

截至 2026-08-25，对当前 `20260821/2_rosbag` 下 18 个现有 bag 的 `metadata.yaml` 检查结果为：实际包含 `base_pnt` 的 bag 数量为 0。该批数据可以继续生成 map 坐标下的数学标定候选，但不能生成可信的绝对 UTM 码牌坐标。绝对 UTM 功能验收必须使用录制脚本更新后新录制且实际包含该话题的数据。

该消息只包含带内坐标，不包含 zone、半球、纬度带或 EPSG。带信息必须由同场地 INSPVAX 经纬度或经审核的设备场地配置提供，不能仅凭 x/y 数值猜测。

### 11.2 坐标换算

按当前坐标约定：

```text
tag_utm_easting  = base_pnt.x + tag_ins_map.x
tag_utm_northing = base_pnt.y + tag_ins_map.y
tag_utm_height   = base_pnt.z + tag_ins_map.z
```

APP/Web 集成层必须确认候选矩阵方向，并使用矩阵平移列中的 `tag_ins_map`。不得把机器人当前 INSPVAX 位置当成码牌位置。

UTM 显示至少包括：

```text
datum: WGS-84
zone_number
latitude_band
hemisphere
epsg
easting_m
northing_m
height_m
coordinate_source
```

北半球 EPSG 为 `32600 + zone`，南半球为 `32700 + zone`。需要显示经纬度时，从最终冻结 UTM 坐标反投影得到，并记录投影库及版本。

### 11.3 一致性和失败处理

APP/Web 或其后端应检查：

- 全部有效 bag 位于同一 UTM 带和同一半球；
- 不同 bag 的 `base_pnt` 差异满足可调门限；
- `base_pnt + ins_map` 与 INSPVAX 投影坐标差异满足可调门限；
- x/y 在合法 UTM 范围内；
- 高度基准语义明确。

相关门限由 APP/Web 配置并进入任务审计，不得隐含在 UI 颜色中。

若 bag 缺少 `base_pnt` 或一致性失败：

- 保留算法已经求得的 map 坐标候选；
- UTM/WGS-84 字段显示为不可用；
- 任务标记坐标审核失败；
- 不得伪造绝对坐标；
- 候选下载入口仍保留。

`base_pnt.z` 与 INSPVAX height 的高程基准需要设备驱动或配置确认；未确认前不得把它标成海拔正高。

## 12. 安全和审计要求

- APP 使用参数数组启动进程，不拼接未转义 shell 字符串；
- 输入 bag 只读，禁止删除和改写；
- 每任务独立输出目录；
- 候选、配置、设备、相机和码牌身份关联保存；
- 保存算法版本、配置 SHA256、候选 ID、训练 bag ID 和失败原因；
- 质量 FAIL 候选不得命名为 `approved`、`installable` 或 `runtime_config`；
- 运行产物、rosbag、日志、CSV、JSON 和 HTML 报告不得提交到代码库；
- 页面下载必须限制在当前任务输出目录内，防止路径越界；
- 一期任何页面和接口都不得执行安装动作。

## 13. 验收标准

### 13.1 算法接口

- 独立包可以构建 wheel；
- 安装后可导入 `dynamic_rosbag_calibration`；
- 运行时不导入旧 `fae_calibration`；
- CLI 仅通过一份配置启动；
- 配置直接接收一组本地 rosbag 路径；
- 重复 ID、路径不存在、缺少 `metadata.yaml` 或重复 YAML key 时明确失败；
- 可调质量门限进入实际运行配置；
- PASS/FAIL 均保留已成功求解候选，执行前异常不伪造候选。

### 13.2 数据录制

- APP 能正确处理 toggle 语义；
- 能显示录制 PID、bag 路径和日志路径；
- 停止后检查 metadata、存储文件和必需话题；
- 同一脚本目录不并行录制；
- 要求 UTM 时确认 bag 实际包含有效 `base_pnt`。

### 13.3 Web/APP 页面

- 每台设备有独立页面；
- 首屏同时显示码牌检测、标定和安装三类状态；
- 检测状态包含证据来源、时间和新鲜度；
- 最近任务状态和历史合格结果状态分开；
- 最近任务 FAIL 仍显示候选和原因；
- 一期安装状态固定 `NOT_INSTALLED`；
- 地图在标定完成后显示固定码牌位置，不随机器人运行更新；
- 新任务失败不覆盖旧有效位置；
- 显示明确的 UTM zone、带内坐标和坐标来源；
- 坐标不可用时明确降级，不显示伪造值。

## 14. 最终约束

本期交付和软件开发应共同遵守：

- 算法侧已交付独立库和唯一配置；
- APP/Web 不再组合旧三份 YAML，也不调用日期脚本；
- 输入是一组有明确角色和路径的 rosbag；
- 当前算法生成多个两-bag候选，不是一次全量联合 B2；
- 质量和审计门限可调，但一次任务内冻结；
- PASS 和 FAIL 均保留真实求解候选；
- 执行异常不得伪造候选；
- 页面独立显示检测、标定和安装状态；
- 最终码牌位置在地图中冻结；
- 绝对坐标按 WGS-84 / UTM 的具体带和带内坐标显示；
- 一期不安装、不覆盖、不切换设备正式运行配置。
