#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PID_FILE="${SCRIPT_DIR}/rosbag_record.pid"
STARTUP_WAIT_SECONDS=1
STOP_TIMEOUT_SECONDS=30

TOPICS=(
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
)

is_valid_pid() {
  [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

is_target_process() {
  local pid="$1"
  local command_line

  kill -0 "$pid" 2>/dev/null || return 1
  command_line="$(ps -p "$pid" -o args= 2>/dev/null)" || return 1
  [[ "$command_line" == *"ros2 bag record"* ]]
}

remove_pid_file() {
  if ! rm -f -- "$PID_FILE"; then
    printf '[ERROR] 无法删除 PID 文件：%s\n' "$PID_FILE" >&2
    return 1
  fi
}

stop_recording() {
  local pid="$1"
  local max_attempts=$((STOP_TIMEOUT_SECONDS * 10))
  local attempt

  printf '[STOP] 正在停止录制，PID：%s\n' "$pid"
  if ! kill -SIGINT "$pid" 2>/dev/null; then
    printf '[ERROR] 无法向录制进程发送 SIGINT，PID 文件予以保留：%s\n' \
      "$PID_FILE" >&2
    return 1
  fi

  for ((attempt = 0; attempt < max_attempts; attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      remove_pid_file || return 1
      printf '[STOP] 录制已停止，PID：%s\n' "$pid"
      return 0
    fi
    sleep 0.1
  done

  printf '[ERROR] 等待 %s 秒后录制进程仍未退出，PID 文件予以保留：%s\n' \
    "$STOP_TIMEOUT_SECONDS" "$PID_FILE" >&2
  return 1
}

start_recording() {
  local timestamp
  local bag_dir
  local log_file
  local recorder_pid
  local temporary_pid_file="${PID_FILE}.tmp.$$"

  if ! command -v ros2 >/dev/null 2>&1; then
    printf '[ERROR] 找不到 ros2 命令，请先加载 ROS 2 环境。\n' >&2
    return 1
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)" || {
    printf '[ERROR] 无法生成录制时间戳。\n' >&2
    return 1
  }
  bag_dir="${SCRIPT_DIR}/bag_${timestamp}"
  log_file="${SCRIPT_DIR}/bag_${timestamp}.log"

  set -m
  nohup ros2 bag record -o "$bag_dir" "${TOPICS[@]}" \
    >"$log_file" 2>&1 < /dev/null &
  recorder_pid=$!
  set +m

  sleep "$STARTUP_WAIT_SECONDS"
  if ! is_target_process "$recorder_pid"; then
    rm -f -- "$temporary_pid_file" "$PID_FILE"
    if ! kill -0 "$recorder_pid" 2>/dev/null; then
      wait "$recorder_pid" 2>/dev/null || true
    fi
    printf '[ERROR] ros2 bag record 启动失败，请查看日志：%s\n' "$log_file" >&2
    return 1
  fi

  if ! printf '%s\n' "$recorder_pid" > "$temporary_pid_file" || \
    ! mv -f -- "$temporary_pid_file" "$PID_FILE"; then
    rm -f -- "$temporary_pid_file"
    if is_target_process "$recorder_pid"; then
      kill -SIGINT "$recorder_pid" 2>/dev/null || true
    fi
    printf '[ERROR] 无法保存 PID 文件：%s\n' "$PID_FILE" >&2
    return 1
  fi

  printf '[START] 录制已启动\n'
  printf '[START] PID：%s\n' "$recorder_pid"
  printf '[START] Bag：%s\n' "$bag_dir"
  printf '[START] Log：%s\n' "$log_file"
}

main() {
  local recorded_pid=''

  if [[ -e "$PID_FILE" ]]; then
    IFS= read -r recorded_pid < "$PID_FILE" || recorded_pid=''

    if is_valid_pid "$recorded_pid" && is_target_process "$recorded_pid"; then
      stop_recording "$recorded_pid"
      return $?
    fi

    if is_valid_pid "$recorded_pid"; then
      printf '[WARN] PID %s 不属于当前 ros2 bag record 进程，正在清理残留状态。\n' \
        "$recorded_pid"
    else
      printf '[WARN] PID 文件内容无效，正在清理残留状态：%s\n' "$PID_FILE"
    fi
    remove_pid_file || return 1
  fi

  start_recording
}

main "$@"
