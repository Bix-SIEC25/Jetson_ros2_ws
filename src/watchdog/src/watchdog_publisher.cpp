// WatchdogPublisher_with_gpu.cpp
#include <chrono>
#include <memory>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <iostream>
#include <vector>
#include <regex>
#include <filesystem>

#include "rclcpp/rclcpp.hpp"
#include "interfaces/msg/log_entry.hpp"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

class WatchdogPublisher : public rclcpp::Node
{
public:
  WatchdogPublisher()
  : Node("watchdog_publisher_gpu"),
    prev_total_(0),
    prev_idle_(0),
    has_prev_cpu_(false)
  {
    publisher_ = this->create_publisher<interfaces::msg::LogEntry>("/logger", 10);

    timer_ = this->create_wall_timer(
      5s, std::bind(&WatchdogPublisher::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "WatchdogPublisher started on jetson");

    logger(2, "jetson_watchdog", "Initializing...");
  }

  void logger(int lvl, const std::string & sender, const std::string & message) {
    auto msg = interfaces::msg::LogEntry();
    msg.level = static_cast<uint8_t>(lvl);
    msg.sender = sender;
    msg.message = message;

    publisher_->publish(msg);
  }

private:
  static bool read_file_first_line(const std::string &path, std::string &out_line) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    std::getline(ifs, out_line);
    // trim
    while (!out_line.empty() && (out_line.back() == '\n' || out_line.back() == '\r')) out_line.pop_back();
    return true;
  }

  bool read_gpu_load_percent(int &gpu_percent) {
    const std::vector<std::string> gpus = {
      "/sys/devices/platform/gpu.0/"//,
      // "/sys/devices/gpu.0/load",
      // "/sys/devices/platform/host1x/gpu.0/load",
      // "/sys/devices/57000000.gpu/load",
      // "/sys/devices/platform/17000000.gv11b/gpu.0/load",
      // "/sys/devices/platform/host1x/17000000.gv11b/load"
    };

    std::string line;
    for (const auto &p : gpus) {
      if (fs::exists(p)) {
        if (read_file_first_line(p, line)) {
          try {
            // remove non-digit characters
            std::string digits;
            for (char c : line) if ((c >= '0' && c <= '9')) digits.push_back(c);
            if (digits.empty()) continue;
            int val = std::stoi(digits);
            gpu_percent = val;
            return true;
          } catch (...) {
            continue;
          }
        }
      }
    }
    return false;
  }

  bool read_gpu_temp_celsius(double &temp_c) {
    const std::string thermal_base = "/sys/class/thermal";
    for (int i = 0; i < 9; ++i) { // there is 9 thermal zones apparently
      std::string type_path = thermal_base + "/thermal_zone" + std::to_string(i) + "/type";
      std::string temp_path = thermal_base + "/thermal_zone" + std::to_string(i) + "/temp";
      std::string ttype;
      if (!fs::exists(type_path) || !fs::exists(temp_path)) continue;
      if (!read_file_first_line(type_path, ttype)) continue;
      std::string ttype_low = ttype;
      for (auto &c : ttype_low) c = std::tolower(c);
      if (ttype_low.find("gpu") != std::string::npos ||
          ttype_low.find("vdd_gpu") != std::string::npos ||
          ttype_low.find("gr3d") != std::string::npos) {
        std::string tstr;
        if (!read_file_first_line(temp_path, tstr)) continue;
        try {
          long val = std::stol(tstr);
          // if value looks like millidegree (1000..100000) divide by 1000
          if (val > 1000) temp_c = static_cast<double>(val) / 1000.0;
          else temp_c = static_cast<double>(val);
          return true;
        } catch (...) {
          continue;
        }
      }
    }
    return false;
  }

  // --- system stats (like cpu... etc) ---

  bool get_disk_usage(const char* path, unsigned long long& total_kb, unsigned long long& available_kb) {
    struct statvfs stat;
    if (statvfs(path, &stat) != 0) {
      return false;
    }
    total_kb = stat.f_blocks * stat.f_frsize / 1024;
    available_kb = stat.f_bavail * stat.f_frsize / 1024;
    return true;
  }

  bool read_cpu_counters(unsigned long long &idle, unsigned long long &total)
  {
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
      return false;
    }
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::string cpu_label;
    iss >> cpu_label;
    if (cpu_label.substr(0,3) != "cpu") {
      return false;
    }

    unsigned long long user=0, nice=0, system=0, idle_t=0, iowait=0, irq=0, softirq=0, steal=0;
    iss >> user >> nice >> system >> idle_t >> iowait >> irq >> softirq >> steal;

    idle = idle_t + iowait;
    total = user + nice + system + idle_t + iowait + irq + softirq + steal;
    return true;
  }

  bool read_meminfo(unsigned long long &mem_total_kb, unsigned long long &mem_available_kb)
  {
    std::ifstream file("/proc/meminfo");
    if (!file.is_open()) {
      return false;
    }
    std::string line;
    mem_total_kb = 0;
    mem_available_kb = 0;
    while (std::getline(file, line)) {
      std::istringstream iss(line);
      std::string key;
      unsigned long long value;
      std::string unit;
      iss >> key >> value >> unit;
      if (key == "MemTotal:") mem_total_kb = value;
      if (key == "MemAvailable:") mem_available_kb = value;
      if (mem_total_kb && mem_available_kb) break;
    }
    return mem_total_kb != 0;
  }

  // 1-minute load average from /proc/loadavg
  bool read_loadavg(double &load1)
  {
    std::ifstream file("/proc/loadavg");
    if (!file.is_open()) return false;
    file >> load1;
    return true;
  }

  bool read_uptime_seconds(double &uptime_seconds)
  {
    std::ifstream file("/proc/uptime");
    if (!file.is_open()) return false;
    file >> uptime_seconds;
    return true;
  }

  bool read_cpu_temp_celsius(double &temp_c)
  {
    std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
    if (!file.is_open()) {
      temp_c = 0.0;
      return false;
    }
    long t_millideg = 0;
    file >> t_millideg;
    if (t_millideg == 0) return false;
    temp_c = static_cast<double>(t_millideg) / 1000.0;
    return true;
  }

  std::string format_uptime(double seconds)
  {
    unsigned long long s = static_cast<unsigned long long>(seconds);
    unsigned long long days = s / 86400ULL;
    s %= 86400ULL;
    unsigned long long hours = s / 3600ULL;
    s %= 3600ULL;
    unsigned long long minutes = s / 60ULL;
    unsigned long long secs = s % 60ULL;

    std::ostringstream oss;
    if (days > 0) oss << days << "d ";
    oss << hours << "h " << minutes << "m " << secs << "s";
    return oss.str();
  }

  // --- timer callback ---
  void timer_callback()
  {
    unsigned long long idle=0, total=0;
    double cpu_percent = -1.0;
    if (read_cpu_counters(idle, total)) {
      if (has_prev_cpu_) {
        unsigned long long totald = total - prev_total_;
        unsigned long long idled = idle - prev_idle_;
        if (totald > 0) {
          double busy = static_cast<double>(totald - idled);
          cpu_percent = (busy / static_cast<double>(totald)) * 100.0;
        } else {
          cpu_percent = 0.0;
        }
      } else {
        // first measurement, no delta
        cpu_percent = -1.0;
        has_prev_cpu_ = true;
      }
      prev_total_ = total;
      prev_idle_ = idle;
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to read /proc/stat");
    }

    // Memory
    unsigned long long mem_total_kb=0, mem_available_kb=0;
    double mem_percent = -1.0;
    if (read_meminfo(mem_total_kb, mem_available_kb)) {
      unsigned long long used_kb = 0;
      if (mem_total_kb > mem_available_kb) used_kb = mem_total_kb - mem_available_kb;
      if (mem_total_kb > 0) {
        mem_percent = (static_cast<double>(used_kb) / static_cast<double>(mem_total_kb)) * 100.0;
      }
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to read /proc/meminfo");
    }

    // Load average
    double load1 = 0.0;
    read_loadavg(load1); // ignore failure

    // Uptime
    double uptime_seconds = 0.0;
    read_uptime_seconds(uptime_seconds);

    // CPU temp
    double cpu_temp_c = 0.0;
    bool has_temp = read_cpu_temp_celsius(cpu_temp_c);

    // GPU stats
    int gpu_util = -1;
    double gpu_temp_c = 0.0;
    bool has_gpu_util = read_gpu_load_percent(gpu_util);
    bool has_gpu_temp = read_gpu_temp_celsius(gpu_temp_c);

    std::ostringstream msg_ss;
    msg_ss << std::fixed << std::setprecision(1);

    if (cpu_percent < 0.0) {
      logger(2, "pi_watchdog", "GeiCar is starting...");
      msg_ss << "CPU: N/A (first reading)";
    } else {
      msg_ss << "CPU: " << cpu_percent << "%";
    }

    if (mem_total_kb > 0) {
      double total_mb = static_cast<double>(mem_total_kb) / 1024.0;
      double used_mb = total_mb * (mem_percent / 100.0);
      msg_ss << " | Mem: " << used_mb << "MiB/" << total_mb << "MiB (" << mem_percent << "%)";
    } else {
      msg_ss << " | Mem: N/A";
    }

    unsigned long long total_disk=0, available_disk=0;
    if (get_disk_usage("/", total_disk, available_disk)) {
      msg_ss << " | Free disk: " << (available_disk / 1024) << "MiB (" << (100 * (total_disk - available_disk)) / total_disk << "%)";
    }

    msg_ss << " | Load1: " << std::setprecision(2) << load1 << std::setprecision(1);

    msg_ss << " | Uptime: " << format_uptime(uptime_seconds);

    if (has_temp) {
      msg_ss << " | Temp: " << cpu_temp_c << "C";
    }

    if (has_gpu_util) {
      msg_ss << " | GPU: " << gpu_util << "%";
    } else {
      msg_ss << " | GPU: N/A";
    }
    if (has_gpu_temp) {
      msg_ss << " | GPU Temp: " << gpu_temp_c << "C";
    }

    // publish as LogEntry
    auto msg = interfaces::msg::LogEntry();
    msg.level = 2;
    msg.sender = "jetson_watchdog";
    msg.message = msg_ss.str();

    publisher_->publish(msg);

    // also log locally
    RCLCPP_INFO(this->get_logger(), "[LEVEL(%u)] %s: %s",
                msg.level, msg.sender.c_str(), msg.message.c_str());
  }

  rclcpp::Publisher<interfaces::msg::LogEntry>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  unsigned long long prev_total_;
  unsigned long long prev_idle_;
  bool has_prev_cpu_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WatchdogPublisher>());
  rclcpp::shutdown();
  return 0;
}
