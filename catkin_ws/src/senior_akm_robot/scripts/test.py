#!/usr/bin/env python
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2


def callback(msg):
    pc = list(pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z")))

    print("Received a point cloud with %d points" % len(pc))
    for i, point in enumerate(pc):
        x, y, z = point
        print(f"NO.{i}:Point coordinates: x={x}, y={y}, z={z}")


if __name__ == "__main__":
    rospy.init_node("test")

    sub = rospy.Subscriber("/point_cloud", PointCloud2, callback)

    rospy.spin()
