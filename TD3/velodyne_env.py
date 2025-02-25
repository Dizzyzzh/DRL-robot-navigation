import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1
LIDAR2ORIGIN_DISTANCE = 0.0887
MAX_PENALTY = 1


# 检查目标位置是否位于障碍物上
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim  # 数据维度
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        # 传感器数据（默认所有方向都是 10m）
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.done = False

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "senior_akm"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # 生成一系列角度区间 (self.gaps)
        # 将 -150° 到 150° 的范围划分成 environment_dim 份，每份的宽度是 π * 4 /  3 / environment_dim
        self.gaps = [[-np.pi / 1.2 - 0.03, -np.pi / 1.2 + np.pi * 4 / 3 / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append([self.gaps[m][1], self.gaps[m][1] + np.pi * 4 / 3 / self.environment_dim])
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # 使用给定的启动文件名启动模拟程序
        rospy.init_node("gym", anonymous=True)

        # 解析 launch 文件路径
        if launchfile.startswith("/"):
            fullpath = launchfile  # 如果是绝对路径，直接使用
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
            # 否则假设 launch 文件存放在当前脚本目录下的 assets 目录中

        # 检查 launch 文件是否存在
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        # 启动 Gazebo 仿真
        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        # 通过 subprocess 启动 roslaunch，指定端口 port 并加载 launch 文件
        print("Gazebo launched!")

        # 设定 ROS 发布者（用于发布控制命令、状态信息等）
        # 机器人速度控制命令发布器
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # 设置 Gazebo 机器人状态的发布器
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        # 控制 Gazebo 物理仿真的服务
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)  # 取消暂停
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)  # 暂停仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)  # 重置 Gazebo 世界
        # Rviz 可视化相关的发布器
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)  # 目标点
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)  # 线速度
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)  # 角速度

        # 设定 ROS 订阅者（用于接收传感器数据、机器人状态等）
        # 订阅 Velodyne 激光雷达点云数据
        self.velodyne = rospy.Subscriber("/point_cloud", PointCloud2, self.velodyne_callback, queue_size=1)
        # 订阅 机器人 里程计数据
        self.odom = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # 读取 velodyne 点云并将其转换为距离数据，返回各障碍物距离
    # range as state representation
    def velodyne_callback(self, v):
        # 读取 Velodyne 激光雷达的点云数据，并转换为列表，每个点包含 (x, y, z) 坐标信息
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))

        # 初始化 self.velodyne_data，假设环境最大检测范围是 10 米
        self.velodyne_data = np.ones(self.environment_dim) * 10

        for i in range(len(data)):
            # 过滤掉地面点，Z 轴相对雷达低于 -0.2 米的点不予考虑
            if data[i][2] > -0.2:
                # 计算该点相对于 X 轴的夹角 β（以机器人坐标系为基准）
                # dot 是该点在 X 轴方向上的投影
                dot = data[i][0] * 1 + data[i][1] * 0

                # 计算该点到原点的距离（即该点的模长）
                mag1 = math.sqrt(math.pow(data[i][0] + LIDAR2ORIGIN_DISTANCE, 2) + math.pow(data[i][1], 2))

                # mag2 是 X 轴方向向量 (1, 0) 的模长，始终等于 1
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))

                try:
                    # 计算该点与 X 轴的夹角 beta，np.sign(data[i][1]) 用于确定角度方向
                    beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                except ValueError:
                    self.done = True
                    break

                # 计算点到原点的欧几里得距离
                # dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)
                dist = math.sqrt((data[i][0] + LIDAR2ORIGIN_DISTANCE) ** 2 + data[i][1] ** 2)

                # 遍历角度区间（self.gaps 存储角度范围的分区，例如 [-π, -π/2], [-π/2, 0] 等）
                for j in range(len(self.gaps)):
                    # 如果当前点的 beta 角度落在某个区间内，则更新对应区域的最小距离
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        # 取该区域内的最小距离值，确保始终记录最近障碍物
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def step(self, action):
        target = False  # 目标是否达成（默认未到达）

        # 发送机器人动作命令（线速度和角速度）
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)  # 可视化机器人轨迹

        # 解除 Gazebo 物理暂停，让机器人执行动作
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        # 让机器人运行一小段时间
        time.sleep(TIME_DELTA)

        # 再次暂停 Gazebo 物理仿真，以便进行状态计算
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # 读取激光雷达数据，并检测是否发生碰撞
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        if self.done:
            done = True
            collision = True
        v_state = self.velodyne_data[:]  # 复制激光雷达数据，避免修改原数据
        laser_state = [v_state]

        # 读取机器人当前位置（里程计数据）
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y

        # 计算机器人朝向角度（从四元数转换为欧拉角）
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)  # 转换为欧拉角
        angle = round(euler[2], 4)  # 提取偏航角（机器人朝向）

        # 计算机器人到目标的距离
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # 计算机器人朝向和目标方向之间的夹角
        skew_x = self.goal_x - self.odom_x  # 目标相对机器人 x 方向
        skew_y = self.goal_y - self.odom_y  # 目标相对机器人 y 方向
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))  # 计算与 x 轴的夹角

        # 处理 beta 角度，使其符合正常范围（-π 到 π）
        if skew_y < 0:
            beta = -beta if skew_x < 0 else 0 - beta

        theta = beta - angle  # 计算机器人朝向与目标方向的误差角度
        # print("theta:", theta)

        # 角度归一化到 [-π, π] 范围内
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # 检查是否到达目标
        if distance < GOAL_REACHED_DIST:
            target = True  # 目标已达成
            done = True  # 终止当前 episode

        # 组合机器人状态（距离、角度误差、线速度、角速度）
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)  # 组合 LiDAR 数据与机器人状态

        # 计算奖励
        reward = self.get_reward(target, collision, action, min_laser, theta, distance)

        # 返回新状态、奖励、是否结束、是否到达目标
        return state, reward, done, target

    def reset(self):
        self.done = False
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            # 重置 Gazebo 世界
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # 设置机器人初始位置
        # 随机生成一个角度，范围为 [-π, π]
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)

        object_state = self.set_self_state
        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            # 随机生成 x 和 y 坐标，范围在 [-4.5, 4.5] 之间
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            # 调用 check_pos 函数检查位置是否有效
            position_ok = check_pos(x, y)

        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        # 保存初始位置的坐标
        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # 设置一个随机的目标位置
        self.change_goal()
        # 随机放置障碍物（如箱子等）
        self.random_box()
        # 发布标记信息
        self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")

        # 复制当前的激光雷达数据
        v_state = []
        v_state[:] = self.velodyne_data[:]
        # 将激光数据包装为 laser_state
        laser_state = [v_state]

        # 计算当前机器人的位置与目标之间的距离
        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

        # 计算目标与机器人位置的偏移量
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        # 计算目标方向与机器人朝向的点积
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))

        # 计算目标方向与机器人朝向之间的夹角
        beta = math.acos(dot / (mag1 * mag2))

        # 根据目标方向的偏移量来调整角度
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        # 计算相对于机器人当前朝向的角度差
        theta = beta - angle

        # 确保角度在 [-π, π] 范围内
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # 组合机器人状态信息，包括目标距离、朝向角度、速度等
        robot_state = [distance, theta, 0.0, 0.0]
        # 将激光数据和机器人状态合并为一个完整的状态
        state = np.append(laser_state, robot_state)

        # 返回初始状态
        return state

    def change_goal(self):
        # 目标范围动态调整：
        # - `self.upper` 和 `self.lower` 分别表示目标生成的上限和下限
        # - 每次调用时，`upper` 增加 0.004，`lower` 减少 0.004
        # - 这样可以在训练过程中逐步扩大目标可能出现的范围，增加难度
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # 在每次重置时随机更改环境中盒子的位置，以随机化训练
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        # 创建用于 Rviz 显示的 MarkerArray
        markerArray = MarkerArray()
        marker = Marker()

        # 设置 Marker 的参考坐标系
        marker.header.frame_id = "/odom"
        marker.type = marker.CYLINDER  # 目标点用圆柱体表示
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01

        # 绿色 (G=1) 表示目标点
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # 目标点的位置
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        # 将 Marker 添加到 MarkerArray，并发布到 Rviz
        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        # ---- 可视化动作（速度） ----

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "/odom"
        marker2.type = marker.CUBE  # 速度用方块表示
        marker2.action = marker.ADD

        # 速度大小决定 x 轴的缩放比例
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01

        # 红色 (R=1) 表示前进速度
        marker2.color.a = 1.0
        marker2.color.r = 0.0
        marker2.color.g = 0.0
        marker2.color.b = 1.0

        # 速度条的位置
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        # ---- 可视化角速度 ----

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "/odom"
        marker3.type = marker.CUBE  # 角速度用方块表示
        marker3.action = marker.ADD

        # 角速度大小决定 x 轴的缩放比例
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01

        # 红色 (R=1) 表示角速度
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0

        # 角速度条的位置
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # 从激光数据中检测碰撞
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    # 计算奖励
    @staticmethod
    def get_reward(target, collision, action, min_laser, theta, distance):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r1 = lambda x: 0 if x < 0 else x
            r2 = lambda x: 0 if x < 0.5 else 0.5 - x
            r3 = lambda x: 1 - x if x < 1 else 0.0
            # return r1(action[0]) - abs(action[1]) / 2 - r3(min_laser) / 2
            # return abs(action[0]) / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
            return r1(action[0]) + r2(abs(theta) / math.pi) - r3(min_laser) / 2  # 效果还行
            # return r2(abs(theta) / math.pi) - r3(min_laser) / 2
            # return -r3(min_laser) / 2
            # return abs(action[0]) / 2 - r3(min_laser) / 2
