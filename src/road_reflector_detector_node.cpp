#include "road_reflector_detector/road_reflector_detector_node.hpp"
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/surface/concave_hull.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace road_reflector_detector {

RoadReflectorDetectorNode::RoadReflectorDetectorNode(const rclcpp::NodeOptions &options)
    : Node("road_reflector_detector_node", options) {
  // Declare parameters
  declare_parameter<std::string>("lidar_topic", "/lidar_points");
  declare_parameter<std::string>("yolo_topic", "/yolo_result");
  declare_parameter<std::string>("camera_info_topic", "/camera_info");
  declare_parameter<std::string>("roi_cloud_topic", "/roi_pointcloud");
  declare_parameter<std::string>("reflector_poses_topic", "/reflector_poses");
  declare_parameter<std::string>("reflector_markers_topic", "/reflector_markers");
  declare_parameter<std::string>("reflector_class_id", "reflector"); //KMS_250626

  // Get parameters
  lidar_topic_ = get_parameter("lidar_topic").as_string();
  yolo_topic_ = get_parameter("yolo_topic").as_string();
  camera_info_topic_ = get_parameter("camera_info_topic").as_string();
  roi_cloud_topic_ = get_parameter("roi_cloud_topic").as_string();
  reflector_poses_topic_ = get_parameter("reflector_poses_topic").as_string();
  reflector_markers_topic_ = get_parameter("reflector_markers_topic").as_string();
  reflector_class_id_ = get_parameter("reflector_class_id").as_string(); //KMS_250626

  // QoS 설정: BestEffort
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();

  // Initialize subscribers with BestEffort QoS
  yolo_sub_ = create_subscription<ultralytics_ros::msg::YoloResult>(
      yolo_topic_, qos, std::bind(&RoadReflectorDetectorNode::yoloCallback, this, std::placeholders::_1));
  lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      lidar_topic_, qos, std::bind(&RoadReflectorDetectorNode::lidarCallback, this, std::placeholders::_1));
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, qos, std::bind(&RoadReflectorDetectorNode::cameraInfoCallback, this, std::placeholders::_1));

  // Initialize publishers with BestEffort QoS
  roi_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(roi_cloud_topic_, qos);
  reflector_pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(reflector_poses_topic_, qos);
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(reflector_markers_topic_, qos);

  // Initialize TF
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  RCLCPP_INFO(this->get_logger(), "Road Reflector Detector Node initialized");
}

void RoadReflectorDetectorNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  camera_info_ = *msg;
  camera_info_received_ = true;
  RCLCPP_INFO(this->get_logger(), "Received camera intrinsic parameters");
}

void RoadReflectorDetectorNode::yoloCallback(const ultralytics_ros::msg::YoloResult::SharedPtr msg) {
  yolo_result_ = msg; // Store YOLO result for use in lidarCallback
  // 디버깅: YOLO 결과 로그 출력
  if (!msg->detections.detections.empty() && !msg->detections.detections[0].results.empty()) {
    RCLCPP_INFO(this->get_logger(), "YOLO detection class_id: %s",
            msg->detections.detections[0].results[0].hypothesis.class_id.c_str());
  }
}

void RoadReflectorDetectorNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  if (!camera_info_received_) {
    RCLCPP_WARN(this->get_logger(), "Camera info not received yet");
    return;
  }

  // Process point cloud with the latest YOLO result
  processPointCloud(msg, yolo_result_);
}

Eigen::Vector4f RoadReflectorDetectorNode::project2Dto3D(float u, float v, float depth) {
  float fx = camera_info_.k[0];
  float fy = camera_info_.k[4];
  float cx = camera_info_.k[2];
  float cy = camera_info_.k[5];

  Eigen::Vector4f point;
  point[0] = (u - cx) * depth / fx;
  point[1] = (v - cy) * depth / fy;
  point[2] = depth;
  point[3] = 1.0;
  return point;
}

void RoadReflectorDetectorNode::processPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg,
                                                 const ultralytics_ros::msg::YoloResult::SharedPtr yolo_msg) {
  if (!yolo_msg || yolo_msg->detections.detections.empty()) {
    RCLCPP_WARN(this->get_logger(), "No YOLO detections available");
    return;
  }

  // Convert ROS PointCloud2 to PCL
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*lidar_msg, *cloud);

  // Transform point cloud to camera frame
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(camera_info_.header.frame_id, lidar_msg->header.frame_id,
                                           lidar_msg->header.stamp);
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Matrix4f tf_matrix;
  tf2::fromMsg(transform.transform, tf_matrix);
  pcl::transformPointCloud(*cloud, *transformed_cloud, tf_matrix);

  // Process each YOLO detection
  for (const auto &detection : yolo_msg->detections.detections) {
    // Use parameterized class_id instead of hardcoded "reflector"
    if (!detection.results.empty() && detection.results[0].hypothesis.class_id != reflector_class_id_) continue; // [주석: KMS_250626]

    // Define 3D ROI based on 2D bounding box
    const auto &bbox = detection.bbox;
    float u_min = bbox.center.position.x - bbox.size_x / 2.0;
    float u_max = bbox.center.position.x + bbox.size_x / 2.0;
    float v_min = bbox.center.position.y - bbox.size_y / 2.0;
    float v_max = bbox.center.position.y + bbox.size_y / 2.0;

    // Assume a reasonable depth range for reflectors (e.g., 1m to 50m)
    float depth_min = 1.0;
    float depth_max = 50.0;

    // Project 2D corners to 3D
    Eigen::Vector4f p1 = project2Dto3D(u_min, v_min, depth_min);
    Eigen::Vector4f p2 = project2Dto3D(u_max, v_max, depth_max);

    // Create crop box
    pcl::CropBox<pcl::PointXYZ> crop_box;
    crop_box.setMin(Eigen::Vector4f(p1[0], p1[1], depth_min, 1.0));
    crop_box.setMax(Eigen::Vector4f(p2[0], p2[1], depth_max, 1.0));
    crop_box.setInputCloud(transformed_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    crop_box.filter(*roi_cloud);

    // Publish ROI cloud for debugging
    sensor_msgs::msg::PointCloud2 roi_msg;
    pcl::toROSMsg(*roi_cloud, roi_msg);
    roi_msg.header = lidar_msg->header;
    roi_cloud_pub_->publish(roi_msg);

    // Cluster ROI cloud
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    clusterPointCloud(roi_cloud, clusters);

    // Process each cluster
    geometry_msgs::msg::PoseArray poses;
    poses.header = lidar_msg->header;
    visualization_msgs::msg::MarkerArray markers;
    int marker_id = 0;

    for (const auto &cluster : clusters) {
      geometry_msgs::msg::Pose pose;
      estimateReflectorPose(cluster, pose);
      poses.poses.push_back(pose);

      // Create visualization marker
      visualization_msgs::msg::Marker marker;
      marker.header = lidar_msg->header;
      marker.ns = "reflectors";
      marker.id = marker_id++;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose = pose;
      marker.scale.x = 0.2;
      marker.scale.y = 0.2;
      marker.scale.z = 0.2;
      marker.color.r = 1.0;
      marker.color.g = 0.0;
      marker.color.b = 0.0;
      marker.color.a = 1.0;
      markers.markers.push_back(marker);
    }

    reflector_pose_pub_->publish(poses);
    marker_pub_->publish(markers);
  }
}

void RoadReflectorDetectorNode::clusterPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                 std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(0.1); // 10cm
  ec.setMinClusterSize(10);
  ec.setMaxClusterSize(1000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  for (const auto &indices : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &idx : indices.indices) {
      cluster->push_back((*cloud)[idx]);
    }
    cluster->width = cluster->size();
    cluster->height = 1;
    cluster->is_dense = true;
    clusters.push_back(cluster);
  }
}

void RoadReflectorDetectorNode::estimateReflectorPose(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster,
                                                     geometry_msgs::msg::Pose &pose) {
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cluster, centroid);

  pose.position.x = centroid[0];
  pose.position.y = centroid[1];
  pose.position.z = centroid[2];
  pose.orientation.w = 1.0; // No rotation
}

}  // namespace road_reflector_detector

RCLCPP_COMPONENTS_REGISTER_NODE(road_reflector_detector::RoadReflectorDetectorNode)