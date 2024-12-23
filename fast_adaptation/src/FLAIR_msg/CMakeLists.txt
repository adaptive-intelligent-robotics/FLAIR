cmake_minimum_required(VERSION 3.8)
project(FLAIR_msg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FLAIRTwist.msg"
  "msg/Perturbation.msg"
  "msg/HumanControl.msg"
  "msg/FunctionalityControllerControl.msg"
  "msg/Latency.msg"
  "msg/GPData.msg"
  "msg/GPMap.msg"
  "msg/Adapted.msg"
  "msg/SystemControl.msg"
  "msg/VnImu.msg"
  DEPENDENCIES geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)

ament_package()
