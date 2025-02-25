#!/usr/bin/env python

import rospy
import laser_geometry.laser_geometry as lg
import sensor_msgs.msg as sensor_msgs


class LaserScanToPointCloud:
    def __init__(self):
        self.lp = lg.LaserProjection()
        self.pc_pub = rospy.Publisher("/point_cloud", sensor_msgs.PointCloud2, queue_size=1)
        self.ls_sub = rospy.Subscriber("/scan", sensor_msgs.LaserScan, self.scan_callback)

    def scan_callback(self, msg):
        # 将LaserScan转换为PointCloud2
        pc2_msg = self.lp.projectLaser(msg)

        # 发布PointCloud2消息
        self.pc_pub.publish(pc2_msg)


if __name__ == "__main__":
    rospy.init_node("laser_scan_to_point_cloud")
    lstopc = LaserScanToPointCloud()
    rospy.spin()
