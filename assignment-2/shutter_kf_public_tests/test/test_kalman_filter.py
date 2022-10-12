#!/usr/bin/env python3
# Public tests for CPSC459/559 Assignment 2 - Part III

PKG = "shutter_kf_public_tests"
NAME = 'test_kalman_filter'

import sys
import unittest

import rospy
import rostest
from std_msgs.msg import Header
from shutter_lookat.msg import Target

from utils import compute_import_path

import_path = compute_import_path('shutter_kf', 'scripts')
sys.path.insert(1, import_path)
from kalman_filter import KalmanFilterNode, KF_predict_step, KF_measurement_update_step

class TestKalmanFilter(unittest.TestCase):
    """
    Public tests for detect_visual_target.py
    """

    def __init__(self, *args):
        """
        Constructor
        """
        super(TestKalmanFilter, self).__init__(*args)
        rospy.init_node(NAME, anonymous=True)

        self.node = KalmanFilterNode.__new__(KalmanFilterNode)

    def create_obs_msg(self):
        x = 1
        y = 8
        z = 1
        obs_msg = Target()
        header = Header()
        header.stamp = rospy.Time.now()
        obs_msg.pose.header = header
        obs_msg.pose.pose.position.x = x
        obs_msg.pose.pose.position.y = y
        obs_msg.pose.pose.position.z = z
        obs_msg.pose.pose.orientation.w = 1
        return obs_msg

    def initialize_node(self):
        self.node.assemble_A_matrix(0.1)
        self.node.assemble_C_matrix()
        self.node.initialize_process_covariance()
        self.node.initialize_measurement_covariance()

    def test_a_assemble_A_matrix(self):
        """
        Callback to that the A matrix has the right dimensions
        """
        self.node.assemble_A_matrix(0.1)
        A = self.node.A
        self.assertTrue(A.shape==(9,9), f"A is not the correct shape. It has shape {A.shape} but it should be 9x9.")

    def test_b_assemble_C_matrix(self):
        """
        Check that the C matrix has the right dimensions
        """
        self.node.assemble_C_matrix()
        C = self.node.C
        self.assertTrue(C.shape==(3,9), f"C is not the correct shape. It has shape {C.shape} but it should be 3x9.")

    def test_c_initialize_process_covariance(self):
        """
        Check that the R matrix has the right dimensions
        """
        self.node.initialize_process_covariance()
        R = self.node.R
        self.assertTrue(R.shape==(9,9), f"R is not the correct shape. It has shape {R.shape} but it should be 9x9.")

    def test_d_initialize_measurement_covariance(self):
        """
        Check that the Q matrix has the right dimensions
        """
        self.node.initialize_measurement_covariance()
        Q = self.node.Q
        self.assertTrue(Q.shape==(3,3), f"Q is not the correct shape. It has shape {Q.shape} but it should be 3x3.")

    def test_e_assemble_observation_vector(self):
        """
        Check that the observation vector is 2D
        """
        obs_msg = self.create_obs_msg()
        z = self.node.assemble_observation_vector(obs_msg)
        self.assertTrue(z.shape==(3,1), f"z is not the right shape. It has shape {z.shape} but it should be 3x1.")

    def test_f_initialize_mu_and_sigma(self):
        """
        Check that the filter is initialized
        """
        obs_msg = self.create_obs_msg()
        self.node.initialize_mu_and_sigma(obs_msg)
        
        mu = self.node.mu
        self.assertTrue(mu.shape==(9,1), f"mu is not the correct shape. It has shape {mu.shape} but it should be 9x1")

        Sigma = self.node.Sigma
        self.assertTrue(Sigma.shape==(9,9), f"Sigma is not the correct shape. It has shape {Sigma.shape} but it should be 9x9")

    def test_g_KF_predict_step(self):
        """
        Check that prediction step function is returning some value
        """
        self.initialize_node()
        obs_msg = self.create_obs_msg()
        self.node.initialize_mu_and_sigma(obs_msg)

        mu, Sigma = KF_predict_step(self.node.mu, self.node.Sigma, self.node.A, self.node.R)

        self.assertIsNotNone(mu, 'mu is None')
        self.assertIsNotNone(Sigma, 'Sigma is not None')

    def test_h_KF_measurement_update_step(self):
        """
        Check that the measurement update function is returning some value
        """
        self.initialize_node()
        obs_msg = self.create_obs_msg()
        self.node.initialize_mu_and_sigma(obs_msg)
        z = self.node.assemble_observation_vector(obs_msg)

        pred_mu, pred_Sigma = KF_predict_step(self.node.mu, self.node.Sigma, self.node.A, self.node.R)
        new_mu, new_Sigma = KF_measurement_update_step(pred_mu, pred_Sigma, z, self.node.C, self.node.Q)

        self.assertIsNotNone(new_mu, 'mu is None')
        self.assertIsNotNone(new_Sigma, 'Sigma is not None')

if __name__ == '__main__':
    rostest.rosrun(PKG, NAME, TestKalmanFilter, sys.argv)
