#!/usr/bin/env python3
import rospy
import numpy as np

from shutter_lookat.msg import Target
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class SimulatedObject(object):
    """
    Simulated object that moves on a circular path in front of the robot.
    The path is contained in a plane parallel to the y-z plane (i.e., x is constant for all points in the path).
    """

    def __init__(self):
        """
        Constructor
        """
        self.x = 1.5                    # x coordinate for the center of the object
        self.center_y = 0.0             # y coordinate for the center of the object's path
        self.center_z = 0.50            # z coordinate for the center of the object's path
        self.angle = 0.0                # current angle for the object in its circular path (relative to the y axis)
        self.radius = 0.5               # radius of the object's circular path
        self.frame = "base_footprint"   # frame in which the coordinates of the object are computed
        self.path_type = "circular"     # type of motion: circular, horizontal, vertical 

    def step(self):
        """
        Update the position of the target based on the publishing rate of the node
        :param publish_rate: node's publish rate
        """
        self.angle += 2.0 * np.pi / 300  # 1 full revolution in 10 secs at 30 Hz

    def get_obj_coord(self):
        if self.path_type == "circular":
            x = self.x
            y = self.center_y + np.sin(self.angle)*self.radius
            z = self.center_z + np.cos(self.angle)*self.radius
        elif self.path_type == 'horizontal':
            x = self.x
            y = self.center_y + np.sin(self.angle)*self.radius
            z = self.center_z 
        elif self.path_type == 'vertical':
            x = self.x
            y = self.center_y 
            z = self.center_z + np.cos(self.angle)*self.radius
        else:
            rospy.logerr(f"Unrecognized path_type for the moving target. Got {self.path_type} but expected 'circular', 'horizontal' or 'vertical'")
            return None
        
        return x,y,z

def generate_target():
    """
    Main function. Publishes the target at a constant frame rate.
    """


    # Init the node
    rospy.init_node('generate_target', anonymous=True)

    # Get ROS params
    x_value = rospy.get_param("~x_value", default=1.5)
    radius = rospy.get_param("~radius", default=0.1)
    publish_rate = rospy.get_param("~publish_rate", default=30)
    path_type = rospy.get_param("~path_type", default="horizontal")
    add_noise = rospy.get_param('~add_noise', False)             # add noise to the target observations? (only set to true for the last part of the assignment!)
    timestamp_buffer = None

    # Create the simulated object
    object = SimulatedObject()
    object.path_type = path_type
    object.x = x_value

    # Define publishers
    vector_pub = rospy.Publisher('/target', Target, queue_size=5)
    marker_pub = rospy.Publisher('/target_marker', Marker, queue_size=5)

    # Publish the target at a constant ratetarget
    rate = rospy.Rate(publish_rate)
    while not rospy.is_shutdown():

        # publish the location of the target as a Vector3Stamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = object.frame
        x, y, z = object.get_obj_coord()
        if add_noise:
           x = x + np.random.normal(loc=0.0, scale=0.02)
           y = y + np.random.normal(loc=0.0, scale=0.01)
           z = z + np.random.normal(loc=0.0, scale=0.01)
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.w = 1.0

        # Check if current Time exceeds clock speed
        if timestamp_buffer is not None and timestamp_buffer >= pose_msg.header.stamp:
            rospy.logwarn('Publish rate exceeds clock speed; check your clock publish rate')
            rate.sleep()
            continue

        target_msg = Target()
        target_msg.pose = pose_msg
        target_msg.radius = radius
        vector_pub.publish(target_msg)
        timestamp_buffer = pose_msg.header.stamp

        # publish a marker to visualize the target in RViz
        marker_msg = Marker()
        marker_msg.header = pose_msg.header
        marker_msg.action = Marker.ADD
        marker_msg.color.a = 0.5
        marker_msg.color.r = 1.0
        marker_msg.lifetime = rospy.Duration(1.0)
        marker_msg.id = 0
        marker_msg.ns = "target"
        marker_msg.type = Marker.SPHERE
        marker_msg.pose = pose_msg.pose
        marker_msg.scale.x = 2.0*radius
        marker_msg.scale.y = 2.0*radius
        marker_msg.scale.z = 2.0*radius
        marker_pub.publish(marker_msg)

        # update the simulated object state
        object.step()

        # sleep to keep the desired publishing rate
        rate.sleep()


if __name__ == '__main__':
    try:
        generate_target()
    except rospy.ROSInterruptException:
        pass