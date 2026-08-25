# dynamic_rosbag_calibration

独立的多 rosbag 动态标定 Python 库。该包从现有 `fae_calibration` 动态标定链路迁移而来，不在运行时导入原工具包，也不负责安装、数据录制、ROS 节点封装、Web 或 APP 交互。

## 输入

唯一公共配置文件为 `config/calibration_job.yaml`。它同时定义 rosbag 角色和绝对路径、机器资产目录、消息定义目录、算法参数、可调质量门限及输出目录。配置中至少需要 2 个 `calibration` rosbag 和 1 个 `validation` rosbag；各 ID 在所有角色间必须唯一。

## 调用

```bash
python3 -m pip install -e .
dynamic-rosbag-calibration --config config/calibration_job.yaml
```

必须通过包安装流程部署依赖；尤其是 `pyapriltags`、`rosbags`、OpenCV、NumPy 和 SciPy。只设置源码 `PYTHONPATH` 但未安装依赖不属于可运行环境，入口会对缺少的 AprilTag 检测器直接报错。

也可以由上层 Python 进程同步调用：

```python
from dynamic_rosbag_calibration import load_job_config, run_calibration

job = load_job_config("config/calibration_job.yaml")
status = run_calibration(str(job.source_path))
```

`load_job_config()` 可用于 Web/APP 在启动任务前做只读校验。`run_calibration()` 返回 0 表示流程正常结束；业务通过/不通过以输出的 `summary.json`、日级报告及候选目录为准。现有算法会保留成功求解的候选文件，质量门限不通过不等同于删除候选。

## 输出

输出位于配置的 `runtime.output_dir`，沿用迁移前动态标定算法的候选、审计、验证和汇总文件结构。运行产物属于本地数据，不应提交到版本库。

数据录制仍使用外部脚本：

```bash
bash /home/share_data/gjf/unloading_using_apriltag_and_pnp/20260821/2_rosbag/rosbag_toggle.sh
```
