#include "io_context.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <thread>

namespace training::io {
namespace {
constexpr double kDeg2Rad = 0.017453292519943295;
constexpr double kRad2Deg = 57.29577951308232;
// 云台俯仰机械限位（仿真器 gimbal_pitch_limit = 0.785 rad，约 45°）。
constexpr double kPitchLimitRad = 0.7853981633974483;
// 反馈超过该时长视为“输入过期”，写指令时按停止处理。
constexpr auto kStaleTimeout = std::chrono::milliseconds(500);

// 消息里的角度制 YXZ 欧拉角 -> 姿态四元数。
// 仿真器按 YXZ 提取姿态：yaw 绕 Y、pitch 绕 X、roll 绕 Z；消息里的 pitch 已取反，故用 -p。
// 正确性在第 4 讲用“正前方靶 -> x 大、y/z≈0”实测校正。
Eigen::Quaterniond imuFromYpr(double yaw_deg, double pitch_deg, double roll_deg) {
  return Eigen::AngleAxisd(yaw_deg * kDeg2Rad, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(-pitch_deg * kDeg2Rad, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(roll_deg * kDeg2Rad, Eigen::Vector3d::UnitZ());
}
}  // namespace

void IoContext::open() {
  if (node_) return;  // 已打开

  context_ = std::make_shared<rclcpp::Context>();
  context_->init(0, nullptr);

  node_ = std::make_shared<rclcpp::Node>("training_io",
                                         rclcpp::NodeOptions().context(context_));

  rclcpp::ExecutorOptions options;
  options.context = context_;
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(options);

  // 图像：仿真器用默认（Reliable）QoS，这里保持一致。
  image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::QoS(10),
      [this](sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_image_ = msg;
      });

  // 云台反馈与指令：仿真器用 SensorDataQoS（BestEffort）。
  recv_sub_ = node_->create_subscription<hnurm_interfaces::msg::VisionRecvData>(
      "/vision_receive_data", rclcpp::SensorDataQoS(),
      [this](hnurm_interfaces::msg::VisionRecvData::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_recv_ = msg;
        last_feedback_ = Clock::now();
      });

  send_pub_ = node_->create_publisher<hnurm_interfaces::msg::VisionSendData>(
      "/vision_send_data", rclcpp::SensorDataQoS());

  executor_->add_node(node_);
}

std::optional<SensorData> IoContext::read() {
  if (!node_) return std::nullopt;
  executor_->spin_some();  // 非阻塞地处理当前已就绪的回调，保证能随主循环退出

  ++read_count_;
  if (read_count_ % 100 == 0) {
    // 演示/验收：每 100 轮打印一次 IO 状态。
    std::lock_guard<std::mutex> lock(mutex_);
    RCLCPP_INFO(node_->get_logger(), "[io] round=%u image=%d feedback=%d", read_count_,
                latest_image_ != nullptr, latest_recv_ != nullptr);
  }

  sensor_msgs::msg::Image::SharedPtr image;
  hnurm_interfaces::msg::VisionRecvData::SharedPtr recv;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // 没有输入，或图像与上次交给算法的是同一帧：不重复交付。
    if (!latest_image_ || !latest_recv_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return std::nullopt;
    }
    const auto& stamp = latest_image_->header.stamp;
    if (delivered_any_ && stamp.sec == last_delivered_stamp_.sec &&
        stamp.nanosec == last_delivered_stamp_.nanosec) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return std::nullopt;
    }
    last_delivered_stamp_ = stamp;
    delivered_any_ = true;
    image = latest_image_;
    recv = latest_recv_;
  }

  // 图像转 BGR，独立拥有像素内存。仿真器固定发布 rgb8。
  if (image->encoding != "rgb8" && image->encoding != "bgr8") return std::nullopt;
  cv::Mat src(image->height, image->width, CV_8UC3,
              const_cast<uint8_t*>(image->data.data()));
  cv::Mat bgr;
  if (image->encoding == "rgb8") {
    cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
  } else {
    bgr = src.clone();
  }

  SensorData data;
  data.image = bgr;
  data.timestamp = to_steady(image->header.stamp);
  data.imu_q = imuFromYpr(recv->yaw, recv->pitch, recv->roll);
  data.bullet_speed = static_cast<double>(recv->bullet_speed.data);

  // 演示/验收：每秒打印一行云台角度与弹速。
  const auto now = Clock::now();
  if (now - last_status_print_ >= std::chrono::seconds(1)) {
    last_status_print_ = now;
    RCLCPP_INFO(node_->get_logger(), "yaw=%.2f deg  pitch=%.2f deg  bullet_speed=%.0f m/s",
                recv->yaw, recv->pitch, data.bullet_speed);
  }

  return data;
}

void IoContext::write(const Command& command) {
  if (!node_ || !send_pub_) return;

  // 安全门控：任何一项不满足都发送停止。
  bool feedback_ok = false;
  bool auto_aim = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    feedback_ok = latest_recv_ != nullptr;
    if (feedback_ok) auto_aim = (latest_recv_->work_mode.data == 0);  // 0 = AUTO_AIM
  }
  const bool stale = (Clock::now() - last_feedback_ > kStaleTimeout);
  const bool finite = std::isfinite(command.yaw) && std::isfinite(command.pitch);
  const bool pitch_ok = std::abs(command.pitch) <= kPitchLimitRad;

  if (!command.control || !feedback_ok || !auto_aim || stale || !finite || !pitch_ok) {
    publish_stop();
    return;
  }

  hnurm_interfaces::msg::VisionSendData msg;
  msg.header.stamp = node_->get_clock()->now();
  msg.header.frame_id = "gimbal_link";
  // 绝对角 rad -> 消息 deg；消息 pitch 方向与 Command 相反。
  msg.yaw = static_cast<float>(command.yaw * kRad2Deg);
  msg.pitch = static_cast<float>(-command.pitch * kRad2Deg);
  // 讲义：当前未提供测距，按协议填 -1。注意仿真器把 -1 视为“输入无效、不控制”，
  // 因此第 5 讲真正要控制云台时需在此填一个 != -1 的真实距离（届时让 Command 携带距离，
  // 或用 lesson04 输出的 position 模长）。
  msg.target_distance = -1.0f;
  msg.target_state.data = 1;  // CONVERGING：控制但不发射
  if (command.shoot && config.allow_fire) {
    msg.target_state.data = 2;  // FIRE：控制并发射
  }
  send_pub_->publish(msg);
}

void IoContext::close() {
  if (!node_) return;
  publish_stop();  // 退出前发送停止指令
  if (executor_) executor_->remove_node(node_);
  context_->shutdown("close");  // 结束回调处理
  node_.reset();
  executor_.reset();
  image_sub_.reset();
  recv_sub_.reset();
  send_pub_.reset();
  context_.reset();
}

void IoContext::publish_stop() {
  if (!node_ || !send_pub_) return;
  hnurm_interfaces::msg::VisionSendData msg;
  msg.header.stamp = node_->get_clock()->now();
  msg.header.frame_id = "gimbal_link";
  msg.target_state.data = 0;   // LOST_TARGET：不控制
  msg.target_type.data = 0;    // NONE
  msg.target_distance = 0.0f;  // 讲义：停止时填 0
  send_pub_->publish(msg);
}

TimePoint IoContext::to_steady(const builtin_interfaces::msg::Time& stamp) const {
  // 把 ROS 系统时间戳映射到程序使用的 steady_clock：二者同速率，仅纪元不同。
  const auto now_steady = Clock::now();
  const auto now_ros = node_->get_clock()->now();  // RCL_SYSTEM_TIME
  const auto latency_ns = (now_ros - rclcpp::Time(stamp)).nanoseconds();
  return now_steady - std::chrono::nanoseconds(latency_ns > 0 ? latency_ns : 0);
}
}  // namespace training::io
