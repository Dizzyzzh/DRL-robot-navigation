#!/usr/bin/env python

# Author: christoph.roesmann@tu-dortmund.de
import rospy, math
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

WHEEL_RADIUS = 0.0625  # 轮胎半径


# 设置最大的角度
def limsteer(data, maxdata):
    if data > 0 and data > maxdata:
        data = maxdata
    elif data < 0 and math.fabs(data) > maxdata:
        data = -maxdata
    return data


def set_senior_akm_velocity_steering(data):
    pub_vel_senior_akm_left_rear_wheel = rospy.Publisher("/wheeltec/senior_akm_left_wheel_velocity_controller/command", Float64, queue_size=1)
    pub_vel_senior_akm_right_rear_wheel = rospy.Publisher("/wheeltec/senior_akm_right_wheel_velocity_controller/command", Float64, queue_size=1)
    pub_vel_senior_akm_left_front_wheel = rospy.Publisher("/wheeltec/senior_akm_left_front_wheel_velocity_controller/command", Float64, queue_size=1)
    pub_vel_senior_akm_right_front_wheel = rospy.Publisher("/wheeltec/senior_akm_right_front_wheel_velocity_controller/command", Float64, queue_size=1)
    pub_pos_senior_akm_left_front_steering_hinge = rospy.Publisher("/wheeltec/senior_akm_left_steering_hinge_position_controller/command", Float64, queue_size=1)
    pub_pos_senior_akm_right_front_steering_hinge = rospy.Publisher("/wheeltec/senior_akm_right_steering_hinge_position_controller/command", Float64, queue_size=1)

    x = data.linear.x
    z = data.angular.z
    L = 0.3187  # 轴距
    T = 0.16  # 两侧轮子之间的距离
    if z != 0 and x != 0:
        r = math.fabs(x / z)  # 转弯半径（车子中心到转弯的圆心）

        rL_rear = r - (math.copysign(1, z) * (T / 2.0))  # r为小车中心的转弯半径，所以T需要除以2在叠加上去
        rR_rear = r + (math.copysign(1, z) * (T / 2.0))
        rL_front = math.sqrt(math.pow(rL_rear, 2) + math.pow(L, 2))
        rR_front = math.sqrt(math.pow(rR_rear, 2) + math.pow(L, 2))
        vL_rear = x * rL_rear / r
        vR_rear = x * rR_rear / r
        vL_front = x * rL_front / r
        vR_front = x * rR_front / r
        anL_front = math.atan2(L, rL_front) * math.copysign(1, z)
        anR_front = math.atan2(L, rR_front) * math.copysign(1, z)

    else:
        vL_rear = x
        vR_rear = x
        vL_front = x
        vR_front = x
        anL_front = z
        anR_front = z

    anL_front = limsteer(anL_front, 0.69813)  # 最大转弯角度的弧度为0.7
    anR_front = limsteer(anR_front, 0.69813)

    pub_vel_senior_akm_left_rear_wheel.publish(vL_rear / WHEEL_RADIUS)
    pub_vel_senior_akm_right_rear_wheel.publish(vR_rear / WHEEL_RADIUS)
    pub_vel_senior_akm_left_front_wheel.publish(vL_front / WHEEL_RADIUS)
    pub_vel_senior_akm_right_front_wheel.publish(vR_front / WHEEL_RADIUS)
    pub_pos_senior_akm_left_front_steering_hinge.publish(anL_front)
    pub_pos_senior_akm_right_front_steering_hinge.publish(anR_front)


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
