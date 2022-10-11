#!/usr/bin/env python3
# Public tests for CPSC459/559 Assignment 2 - Part II

PKG = "shutter_kf_public_tests"
NAME = 'test_filter_quality'

import sys
import unittest
import numpy as np

import rospy
import rostest

import message_filters
from geometry_msgs.msg import PoseStamped

class TestFilterQuality(unittest.TestCase):
    """
    Public tests for kalman_filter.py
    """

    def __init__(self, *args):
        """
        Constructor
        """
        super(TestFilterQuality, self).__init__(*args)

        self.filtered_topic = "/filtered_target"    # filtered topic
        self.gt_topic = "/true_target"              # true topic
        self.msg_list = []

        rospy.init_node(NAME, anonymous=True)


    def _callback(self, filtered_msg, gt_msg):
        self.msg_list.append((filtered_msg, gt_msg))
    

    def test_quality(self):
        """
        Check that the information is being published on /virtual_camera/camera_info
        """
        rospy.sleep(2)

        self.msg_list = []

        sub1 = message_filters.Subscriber(self.filtered_topic, PoseStamped)
        sub2 = message_filters.Subscriber(self.gt_topic, PoseStamped)
        subs = [sub1, sub2]
        ts = message_filters.ApproximateTimeSynchronizer(subs, 10, 0.25)
        ts.registerCallback(self._callback)
        
        timeout_t = rospy.Time.now() + rospy.Duration.from_sec(40)  # 20 seconds in the future

        # wait patiently for a message
        while not rospy.is_shutdown():
            rospy.sleep(0.1)
            if rospy.Time.now() > timeout_t:
                break

        # stop getting messages
        for s in subs:
            s.sub.unregister()

        # did we succeed in getting messages?
        self.assertGreaterEqual(len(self.msg_list), 10, 
            f"Got less than 10 sincronized message in 20 secs (num. messages={len(self.msg_list)}).")

        # compute error
        err = []
        for i in range(len(self.msg_list)):
            m1 = self.msg_list[i][0]
            m2 = self.msg_list[i][1]
            diffx = m1.pose.position.x - m2.pose.position.x
            diffy = m1.pose.position.y - m2.pose.position.y
            diffz = m1.pose.position.z - m2.pose.position.z
            l2_err = np.sqrt(diffx*diffx + diffy*diffy + diffz*diffz)
            err.append(l2_err)

        avg_err = np.mean(err)
        std_err = np.std(err)
        self.assertLessEqual(avg_err, 0.5, 
            f"The average error {avg_err} (+- {std_err}) was greater than 0.5.")


if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestFilterQuality, sys.argv)
