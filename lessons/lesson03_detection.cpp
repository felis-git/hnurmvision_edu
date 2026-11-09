#include "lessons.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace training {
namespace {
// ============================================================================
// 第 3 讲：装甲板检测。
//
// 思路（经典 RoboMaster 灯条检测）：
//   1. 敌方颜色分割：仿真器己方固定红方、敌方恒为蓝方，故只检测蓝色；
//   2. 灯条 = “蓝色” 且 “高亮”（LED 灯条比车身蓝得更亮，用来和车身区分）；
//   3. 形态学清理 → 轮廓 → minAreaRect 提取细长条状的灯条；
//   4. 左右两条灯条按几何约束配对成一块装甲板；
//   5. 由两条灯条的外沿 4 角点给出板框（左上、右上、右下、左下）。
//
// 所有阈值都集中在下面，方便按实际仿真画面标定。
// ============================================================================

// ---- 颜色与亮度（OpenCV HSV，H∈[0,179]，S/V∈[0,255]）----
constexpr int kBlueHueLow = 90;    // 蓝色色相下界
constexpr int kBlueHueHigh = 135;  // 蓝色色相上界（红色约 0~10、170~179，被排除）
constexpr int kBlueSatLow = 60;    // 饱和度下界：把白/灰/暗部排除
constexpr int kBlueValLow = 60;    // 蓝色通道亮度下界（先取整块蓝，再与亮掩码相交）
constexpr int kBrightValLow = 200; // 高亮下界：灯条比车身更亮，用来分离灯条

// ---- 灯条几何 ----
constexpr double kMinBarArea = 20.0;       // 灯条最小面积（px²），滤噪
constexpr double kMinBarAspect = 2.0;      // 灯条长/宽最小比（细长条状）
constexpr double kMaxBarTiltDeg = 30.0;    // 灯条相对竖直的最大倾角

// ---- 配对成板 ----
constexpr double kMaxLenRatio = 1.6;       // 左右灯条长度比上限
constexpr double kMaxCenterYDiff = 0.6;    // 左右灯条中心 y 差 / 平均长度
constexpr double kMinPairDist = 0.8;       // 左右灯条水平间距 / 平均长度（下限）
constexpr double kMaxPairDist = 4.5;       // 同上（上限）
constexpr double kMaxTiltDiffDeg = 15.0;   // 左右灯条倾角差上限

// ---- 大小板 ----
// 板宽 / 灯条高 大于该值判为大板。真实小装甲约 135/55≈2.45，大装甲约 230/55≈4.2。
constexpr double kLargeArmorRatio = 3.2;

// 调试开关：置 true 后克隆图像并画出灯条/板框，便于标定阈值；交付前改回 false。
constexpr bool kDebugDraw = true;

// 一条灯条的图像观测。
struct LightBar {
  cv::Point2f tl, tr, br, bl;  // 灯条自身 4 角（图像坐标）
  cv::Point2f center;          // 灯条中心
  double length;               // 灯条高（长边，px）
  double width;                // 灯条宽（短边，px）
  double tilt_deg;             // 相对竖直的倾角，0=竖直
};

// 由 minAreaRect 的四顶点整理出 tl/tr/br/bl，并计算长度/宽度/倾角。
// 图像坐标 y 向下，故竖直向下的方向向量为 (0,+1)。
LightBar makeLightBar(const cv::RotatedRect& r) {  // 自带函数 minAreaRect 得到旋转矩形 RotatedRect
  LightBar b{};  // 创建对象b，并初始化为0
  cv::Point2f pts[4];
  r.points(pts);  // RotatedRect::points 直接把 4 个顶点写入数组

  // 按 y 分成上边两点、下边两点。
  std::sort(pts, pts + 4,
            [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
  // sort( 起始地址，终止地址，Lambda表达式[](接收参数){排序核心规则} );
  // Lambda表达式 匿名小函数 用于简单只使用一次的排序规则，省去定义步骤
  // [] 身份证，告诉用到匿名函数
  b.tl = (pts[0].x <= pts[1].x) ? pts[0] : pts[1];  // 上边 x 小者
  b.tr = (pts[0].x <= pts[1].x) ? pts[1] : pts[0];  // 上边 x 大者
  b.bl = (pts[2].x <= pts[3].x) ? pts[2] : pts[3];  // 下边 x 小者
  b.br = (pts[2].x <= pts[3].x) ? pts[3] : pts[2];  // 下边 x 大者

  const cv::Point2f top = (b.tl + b.tr) * 0.5f;
  const cv::Point2f bot = (b.bl + b.br) * 0.5f;
  b.center = (top + bot) * 0.5f;

  const cv::Point2f d = bot - top;                 // 从上到下的长轴方向
  b.length = std::sqrt(d.x * d.x + d.y * d.y);
  b.width = 0.5 * (cv::norm(b.tr - b.tl) + cv::norm(b.br - b.bl));
  b.tilt_deg = std::atan2(d.x, d.y) * 180.0 / kPi;  // atan2(水平, 竖直)，0=竖直
  return b;
}

// 尝试把两条灯条配成一块板。成功返回 Armor，否则返回 std::nullopt。
std::optional<Armor> tryPair(const LightBar& a, const LightBar& b) {
  // 保证 left 在左（x 小）、right 在右（x 大）。
  const LightBar* left = &a;
  const LightBar* right = &b;
  if (left->center.x > right->center.x) std::swap(left, right);  
  // 这里swap不能直接交换a，b  a，b被设为const无法修改内容  所以引入指针

  const double len = 0.5 * (left->length + right->length);
  const double dist = right->center.x - left->center.x;      // > 0
  const double dy = std::abs(left->center.y - right->center.y);
  const double dtilt = std::abs(left->tilt_deg - right->tilt_deg);
  const double lenRatio =
      std::max(left->length, right->length) / std::min(left->length, right->length);

  if (lenRatio > kMaxLenRatio) return std::nullopt;  //nullopt空值状态  不是nullptr空指针
  if (dy > kMaxCenterYDiff * len) return std::nullopt;
  if (dist < kMinPairDist * len || dist > kMaxPairDist * len) return std::nullopt;
  if (dtilt > kMaxTiltDiffDeg) return std::nullopt;

  Armor armor;
  // 板框四角：左灯条外沿给出左上/左下，右灯条外沿给出右上/右下。
  armor.corners = {left->tl, right->tr, right->br, left->bl};
  armor.large_armor = (dist / len) > kLargeArmorRatio;
  // 编号识别（OCR/模板匹配）留作扩展；认不出按约定填 -1。
  armor.class_id = -1;

  // 置信度：由长度一致性、倾角一致性综合得到，0~1。 0~1 是置信度行业通用标准
  const double confLen = 1.0 - (lenRatio - 1.0) / (kMaxLenRatio - 1.0);
  const double confTilt = 1.0 - 0.5 * (dtilt / kMaxTiltDiffDeg);
  armor.confidence = std::clamp(0.5 * confLen + 0.5 * confTilt, 0.0, 1.0);
  // clamp 钳位 将数值限制在最大值和最小值之间 clamp(数值，最小值，最大值);
  return armor;
}
}  // namespace

std::vector<Armor> lesson03_detection(const cv::Mat& image) {
  if (image.empty()) return {};

  // ---- 1. 蓝色 + 高亮 → 灯条候选掩码 ----
  cv::Mat hsv;  //HSV颜色空间 色相，饱和度，亮度
  cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);  
  // cv::COLOR_BGR2HSV 转换模式
  // cv::cvtColor() 转换颜色空间函数

  cv::Mat blue, bright, mask;  // mask 存两者交集
  cv::inRange(hsv, cv::Scalar(kBlueHueLow, kBlueSatLow, kBlueValLow),
              cv::Scalar(kBlueHueHigh, 255, 255), blue);
  cv::inRange(hsv, cv::Scalar(0, 0, kBrightValLow), cv::Scalar(179, 255, 255), bright);
  // cv::inRange() 用于提取特定颜色范围
  // cv::inRange(输入图像，下限Scalar{H，S，V下限}，上限Scalar{H，S，V上限}，输出图像);
  cv::bitwise_and(blue, bright, mask);
  // cv::bitwise_and() 求交集  cv::bitwise_or() 求并集

  // ---- 2. 形态学清理：先闭合弥合灯条断裂，再开运算去噪 ----
  // 闭运算：填补缝隙，修补  开运算：去处毛刺，清理
  const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  // cv::getStructuringElement() 创建用于形态学操作的核 “小刷子”  cv::MORPH_RECT 矩形刷子
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  // cv::morphologyEx() 高级形态学转换函数

  // ---- 3. 提取灯条 ----
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<LightBar> bars;
  bars.reserve(contours.size());  // reserve 为数组预分配内存
  
  for (const auto& c : contours) {
    if (cv::contourArea(c) < kMinBarArea) continue;  // cv::contourArea 计算轮廓面积 
    LightBar bar = makeLightBar(cv::minAreaRect(c));  // cv::minAreaRect 计算轮廓最小包围旋转矩形
    if (bar.length < 1.0) continue;
    if (bar.length / bar.width < kMinBarAspect) continue;
    if (std::abs(bar.tilt_deg) > kMaxBarTiltDeg) continue;
    bars.push_back(bar);
  }

  // ---- 4. 配对成板（按质量贪心，每条灯条最多用一次）----
  struct Pair {
    int i, j;
    double score;
  };
  
  std::vector<Pair> pairs;
  for (int i = 0; i < static_cast<int>(bars.size()); ++i) {  //i++会备份旧值
    for (int j = i + 1; j < static_cast<int>(bars.size()); ++j) {
  // bars.size()返回类型为 无符号整数 size_t
  // static_cast<int>() 将其强制转换为 有符号整数 int（新式“静态类型转换“）
  // 相比于旧式 (int)bars.size() 更安全
      auto armor = tryPair(bars[i], bars[j]);
      if (!armor) continue;
      pairs.push_back({i, j, armor->confidence});
    }
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b) { return a.score > b.score; });

  std::vector<Armor> armors;
  std::vector<bool> used(bars.size(), false);
  for (const auto& p : pairs) {
    if (used[p.i] || used[p.j]) continue;  // 链式赋值
    used[p.i] = used[p.j] = true;
    armors.push_back(*tryPair(bars[p.i], bars[p.j]));
    // optional重载了 * ，写 * 时是打开optional这个盒子取出真实对象 
  }

  // ---- 5. 调试显示（默认关闭）----
  if (kDebugDraw) {
    static int frame = 0;
    if (++frame % 30 == 0) {  // 每一张图片就是一帧
      std::cerr << "[lesson03] bars=" << bars.size() << " armors=" << armors.size()
                << std::endl;
    }
    cv::Mat vis = image.clone();
    for (const auto& b : bars) {
      std::vector<cv::Point> poly = {b.tl, b.tr, b.br, b.bl};
      cv::polylines(vis, poly, true, cv::Scalar(0, 255, 255), 1);  // 灯条：黄色
    }
    for (const auto& a : armors) {
      std::vector<cv::Point> poly(a.corners.begin(), a.corners.end());
      cv::polylines(vis, poly, true, cv::Scalar(0, 255, 0), 2);    // 板框：绿色
      // 绘制多边形轮廓 cv::polylines(画板，路径点（容器，按填入顺序连线），是否闭合，颜色（BGR格式），线条粗细（1或2）)；
    }
    cv::imshow("lesson03_debug", vis);  
    // 弹出窗口，显示图像 cv::imshow(“窗口名字”，显示图像)；
    cv::waitKey(1);
    // 带键盘监听功能的短暂延时函数 单位 ms
  }

  return armors;
}
}  // namespace training
