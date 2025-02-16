#!/usr/bin/env python

# Author: christoph.roesmann@tu-dortmund.de
import rospy, math
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

WHEEL_DIAMETER = 0.125  # 轮胎直径
WHEEL_RADIUS = WHEEL_DIAMETER / 2  # 轮胎半径
WHEEL_BASE = 0.3187  # 轴距
TRACK_WIDTH = 0.16  # 轮距


def set_senior_akm_velocity_steering(data):
    pub_vel_senior_akm_left_rear_wheel = rospy.Publisher("/wheeltec/senior_akm_left_wheel_velocity_controller/command", Float64, queue_size=1)
    pub_vel_senior_akm_right_rear_wheel = rospy.Publisher("/wheeltec/senior_akm_right_wheel_velocity_controller/command", Float64, queue_size=1)
    pub_pos_senior_akm_left_front_steering_hinge = rospy.Publisher(
        "/wheeltec/senior_akm_left_steering_hinge_position_controller/command", Float64, queue_size=1
    )
    pub_pos_senior_akm_right_front_steering_hinge = rospy.Publisher(
        "/wheeltec/senior_akm_right_steering_hinge_position_controller/command", Float64, queue_size=1
    )
    # cmd_vel linear.x
    # 线速度
    v = data.linear.x
    # cmd_vel angular.z
    # 角速度
    w = data.angular.z

    # akm velocity calculation
    # 判断前进后退
    if v > 0:
        g = 1
    elif v == 0:
        g = 0
    else:
        g = -1

    # 判断左转右转
    if w >= 0:
        k = 1  # turn left
    else:
        k = -1  # turn right

    if w == 0:
        wl = 0
        wr = 0
        # 计算后轮转速
        vlr = v / (WHEEL_RADIUS)  # w=v/r moving forward
        vrr = v / (WHEEL_RADIUS)
    else:
        R = v / w  # 转弯半径
        a2 = TRACK_WIDTH / 2
        m = R**2 - a2**2
        ml = abs(m) ** 0.5 - k * (TRACK_WIDTH / 2)
        mr = abs(m) ** 0.5 + k * (TRACK_WIDTH / 2)

        if m < 0 or ml < 0 or mr < 0:
            wl = 0
            wr = 0
            vlr = 0
            vrr = 0
        else:
            vlr = (g * (((v**2) - ((w * a2) ** 2)) ** 0.5 - ((g * TRACK_WIDTH * w) / 2))) / (WHEEL_DIAMETER / 2)
            vrr = (g * (((v**2) - ((w * a2) ** 2)) ** 0.5 + ((g * TRACK_WIDTH * w) / 2))) / (WHEEL_DIAMETER / 2)
            wl = g * k * math.atan(WHEEL_BASE / ml)
            wr = g * k * math.atan(WHEEL_BASE / mr)

    # publish velocity to controller command
    pub_pos_senior_akm_left_front_steering_hinge.publish(wl)
    pub_pos_senior_akm_right_front_steering_hinge.publish(wr)
    pub_vel_senior_akm_left_rear_wheel.publish(vlr)
    pub_vel_senior_akm_right_rear_wheel.publish(vrr)


def Sub_cmd_vel_senior_akm():

    rospy.init_node("Sub_cmd_vel_senior_akm", anonymous=True)
    # Subscriber cmd_vel
    rospy.Subscriber("/cmd_vel", Twist, set_senior_akm_velocity_steering, queue_size=1)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == "__main__":
    try:
        Sub_cmd_vel_senior_akm()
    except rospy.ROSInterruptException:
        pass
