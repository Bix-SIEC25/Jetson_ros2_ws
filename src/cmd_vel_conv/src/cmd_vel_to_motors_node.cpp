#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "interfaces/msg/motors_order.hpp"

class CmdVelToMotors : public rclcpp::Node
{
public:
  CmdVelToMotors()
  : Node("cmd_vel_to_motors_node")
  {
    // Paramètres : vitesses max utilisées pour normaliser
    max_linear_speed_  = declare_parameter<double>("max_linear_speed", 1.0);   // [m/s]
    max_angular_speed_ = declare_parameter<double>("max_angular_speed", 1.0);  // [rad/s]

    motors_pub_ = create_publisher<interfaces::msg::MotorsOrder>("motors_order_raw", 10);

    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10,
      std::bind(&CmdVelToMotors::cmdVelCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "cmd_vel_to_motors_node READY");
  }

private:
  static double clamp(double x, double min_val, double max_val)
  {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
  }

  void cmdVelCallback(const geometry_msgs::msg::Twist & msg)
  {
    auto out = interfaces::msg::MotorsOrder();

    // ----- 1) Gestion de la translation avant / arrière -----
    double v = msg.linear.x;  // m/s, on considère que Nav2 reste dans [-max_linear_speed_, max_linear_speed_]

    v = clamp(v, -max_linear_speed_, max_linear_speed_);

    bool reverse = (v < 0.0);
    double v_norm = 0.0; // [0;1]

    if (max_linear_speed_ > 1e-3) {
      v_norm = std::abs(v) / max_linear_speed_;   // 0 à 1
    }
    v_norm = clamp(v_norm, 0.0, 1.0);

    uint8_t pwm;
    if (v_norm < 1e-3) {
      // Stop
      pwm = 50;
    } else if (!reverse) {
      // Avant : 50 -> 100
      pwm = static_cast<uint8_t>(50.0 + 50.0 * v_norm)*1.4;
    } else {
      // Arrière : 50 -> 0
      pwm = static_cast<uint8_t>(50.0 - 50.0 * v_norm)*10.0;
    }

    out.left_rear_pwm  = static_cast<int8_t>(pwm);
    out.right_rear_pwm = static_cast<int8_t>(pwm);

    // ----- 2) Gestion de la rotation (steering) -----
    double w = msg.angular.z;  // rad/s
    w = clamp(w, -max_angular_speed_, max_angular_speed_);

    double steer_norm = 0.0; // [-1 ; 1]
    if (max_angular_speed_ > 1e-3) {
      steer_norm = -(w / max_angular_speed_)*1.4;
    }
    steer_norm = clamp(steer_norm, -1.0, 1.0);

    // Même convention que car_control_node : [-1,1] -> [-127,+127]
    out.steering_pwm = static_cast<int8_t>(steer_norm * 100.0);

    motors_pub_->publish(out);
  }

  // Params
  double max_linear_speed_;
  double max_angular_speed_;

  // ROS
  rclcpp::Publisher<interfaces::msg::MotorsOrder>::SharedPtr motors_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CmdVelToMotors>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
