cmake_minimum_required(VERSION 3.8)
project(road_reflector_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Suppress PCL_ROOT warning
cmake_policy(SET CMP0074 NEW)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)
find_package(std_msgs REQUIRED) # YoloResult.msg 의존성
find_package(vision_msgs REQUIRED) # YoloResult.msg 의존성

# Manually specify ultralytics_ros paths
set(ultralytics_ros_INCLUDE_DIRS ~/workspace/ultralytics_ws/install/ultralytics_ros/include/ultralytics_ros)
# ultralytics_ros의 라이브러리가 없으므로 생략 (필요 시 추가)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${ultralytics_ros_INCLUDE_DIRS}
)

# Add component library
add_library(road_reflector_detector_component SHARED
  src/road_reflector_detector_node.cpp
)

# Link libraries
target_link_libraries(road_reflector_detector_component
  ${PCL_LIBRARIES}
)

# Specify dependencies
ament_target_dependencies(road_reflector_detector_component
  rclcpp
  rclcpp_components
  sensor_msgs
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  vision_msgs
)

# Register component
rclcpp_components_register_node(road_reflector_detector_component
  PLUGIN "road_reflector_detector::RoadReflectorDetectorNode"
  EXECUTABLE road_reflector_detector_node
)

# Install
install(TARGETS road_reflector_detector_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  launch config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()