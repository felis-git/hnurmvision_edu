#pragma once
#include "config.hpp"
#include "types.hpp"

#include <builtin_interfaces/msg/time.hpp>
#include <hnurm_interfaces/msg/vision_recv_data.hpp>
#include <hnurm_interfaces/msg/vision_send_data.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <mutex>
#include <optional>

namespace training::io {
// 第 2 讲：仿真 IO。对外只暴露四个方法；对内负责 ROS2 通信、时间对齐与安全门控。
struct IoContext {
  Config config;

  explicit IoContext(Config c) : config(c) {}

  void open();
  std::optional<SensorData> read();
  void write(const Command& command);
  void close();

 private:
  void publish_stop();
  // 把 ROS 系统时间戳映射到程序使用的 steady_clock。
  TimePoint to_steady(const builtin_interfaces::msg::Time& stamp) const;

  rclcpp::Context::SharedPtr context_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<hnurm_interfaces::msg::VisionRecvData>::SharedPtr recv_sub_;
  rclcpp::Publisher<hnurm_interfaces::msg::VisionSendData>::SharedPtr send_pub_;

  // 单线程 Executor 使回调与 read() 在同一线程串行执行；mutex 仅作防御。
  std::mutex mutex_;
  sensor_msgs::msg::Image::SharedPtr latest_image_;
  hnurm_interfaces::msg::VisionRecvData::SharedPtr latest_recv_;

  // 已交给算法的最新图像时刻，用于“同一图像只交一次”的去重。
  builtin_interfaces::msg::Time last_delivered_stamp_{};
  bool delivered_any_ = false;
  // 最近一次反馈到达时刻，用于“输入过期”判断；默认纪元，未收到反馈前视为过期。
  Clock::time_point last_feedback_{};

  // 演示/验收辅助。
  Clock::time_point last_status_print_{};
  unsigned read_count_ = 0;
};
}  // namespace training::io
