#ifndef ROAD_REFLECTOR_DETECTOR_NODE_HPP_
#define ROAD_REFLECTOR_DETECTOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <ultralytics_ros/msg/yolo_result.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/extract_clusters.h>
#include <memory>

namespace road_reflector_detector {

class RoadReflectorDetectorNode : public rclcpp::Node {
public:
  explicit RoadReflectorDetectorNode(const rclcpp::NodeOptions &options);

private:
  // Subscribers
  rclcpp::Subscription<ultralytics_ros::msg::YoloResult>::SharedPtr yolo_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr roi_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr reflector_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // TF
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

  // Camera intrinsic parameters
  sensor_msgs::msg::CameraInfo camera_info_;
  bool camera_info_received_{false};

  // YOLO result storage
  ultralytics_ros::msg::YoloResult::SharedPtr yolo_result_;

  // Parameters
  std::string lidar_topic_;
  std::string yolo_topic_;
  std::string camera_info_topic_;
  std::string roi_cloud_topic_;
  std::string reflector_poses_topic_;
  std::string reflector_markers_topic_;
  std::string reflector_class_id_; //KMS_250626

  // Callbacks
  void yoloCallback(const ultralytics_ros::msg::YoloResult::SharedPtr msg);
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  // Processing functions
  void processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg,
                        const ultralytics_ros::msg::YoloResult::SharedPtr yolo_msg);
  Eigen::Vector4f project2Dto3D(float u, float v, float depth);
  void clusterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters);
  void estimateReflectorPose(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
                           geometry_msgs::msg::Pose &pose);
};

}  // namespace road_reflector_detector

#endif  // ROAD_REFLECTOR_DETECTOR_NODE_HPP_