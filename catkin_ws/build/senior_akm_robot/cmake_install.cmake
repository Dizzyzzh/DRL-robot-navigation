# Install script for directory: /home/ubuntu/Code/DRL-robot-navigation/catkin_ws/src/senior_akm_robot

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/senior_akm_robot.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/senior_akm_robot/cmake" TYPE FILE FILES
    "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/senior_akm_robotConfig.cmake"
    "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/senior_akm_robotConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/senior_akm_robot" TYPE FILE FILES "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/src/senior_akm_robot/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/senior_akm_robot" TYPE PROGRAM FILES "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/Sub_cmd_vel_senior_akm.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/senior_akm_robot" TYPE PROGRAM FILES "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/real_cmd_vel_senior_akm.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/senior_akm_robot" TYPE PROGRAM FILES "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/LaserScan_to_PointCloud.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/senior_akm_robot" TYPE PROGRAM FILES "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/gazebo_odometry.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/senior_akm_robot" TYPE PROGRAM FILES "/home/ubuntu/Code/DRL-robot-navigation/catkin_ws/build/senior_akm_robot/catkin_generated/installspace/test.py")
endif()

