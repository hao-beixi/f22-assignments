#!/usr/bin/env python3

import copy
import queue
import cv2
import numpy as np
import rospy
from shutter_lookat.msg import Target
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray

from threading import Lock


def KF_predict_step(mu, Sigma, A, R):
    """
    Prediction step for the kalman filter
    :param mu: prior state mean
    :param Sigma: prior state covariance
    :param A: linear transition function for the state
    :param R: covariance for the noise of the transition model
    :return: predicted mean and covariance for the state based on the transition model A and the process noise R.
    """
    # TODO. Complete. Set the predicted_mu and predicted_Sigma variables according to the Kalman Filter's prediction step.
    return predicted_mu, predicted_Sigma


def KF_measurement_update_step(pred_mu, pred_Sigma, z, C, Q):
    """
    Correction step for the kalman filter
    :param pred_mu: predicted mean for the state (from KF_predict_step). Should be a 6x1 vector.
    :param pred_Sigma: predicted covariance for the state (from KF_predict_step). Should be a 6x6 matrix.
    :param z: measurement.
    :param C: matrix that transforms the state into measurement by Cx.
    :param Q: covariance for the noise of the observation or measurement model.
    :return: corrected mean and covariance for the state based on the observation z, the linear model C, and the measurement cov. Q.
    """
    # TODO. Complete. Set the corrected_mu and corrected_Sigma variables according to the Kalman Filter's measurement update step.
    return corrected_mu, corrected_Sigma


class KalmanFilterNode(object):
    """Kalman Filter node"""

    def __new__(cls):
        return super(KalmanFilterNode, cls).__new__(cls)

    def __init__(self):
        """
        Constructor
        """

        # Init the node
        rospy.init_node('kalman_filter_node')

        # Note parameters
        frame_rate = rospy.get_param("~frame_rate", 20)        # fps
        future_steps = rospy.get_param("~future_steps", 20)    # how many steps to make a prediction into the future

        # Filter variables
        self.mu = None                                     # state mean
        self.Sigma = None                                  # state covariance
        self.R = None                                      # covariance for the process model
        self.Q = None                                      # covariance for the measurement model
        self.A = None                                      # matrix that predicts the new state based on the prior state
        self.C = None                                      # matrix that transforms states into observations

        # Initialize constant filter values
        self.initialize_process_covariance()
        self.initialize_measurement_covariance()
        self.assemble_C_matrix()

        # other node variables
        self.frame_id = None                               # frame id for the observations and filtering
        self.latest_observation_msg = None                 # latest PointStamped observation that was received by the node
        self.mutex = Lock()                                # mutex for the observation
        self.bridge = CvBridge()                           # OpenCV - ROS bridge
        self.observed_positions = queue.Queue(50)          # FIFO queue of observed positions (for visualization)
        self.tracked_positions = queue.Queue(50)           # FIFO queue of tracked positions (for visualization)
        last_time = None                                   # Last time-stamp of when a filter update was done

        # Publishers
        self.pub_filtered = rospy.Publisher("/filtered_target", PoseStamped, queue_size=5)
        self.pub_future = rospy.Publisher("/future_target", PoseStamped, queue_size=5)
        self.pub_markers = rospy.Publisher("/filtered_markers", MarkerArray, queue_size=5)

        # Subscribers
        self.obs_sub = rospy.Subscriber("/target", Target, self.obs_callback, queue_size=5)

        # Main loop
        while not rospy.is_shutdown():

            # publish visualization info on every iteration
            self.publish_position_markers()

            # copy the last received message for the filter update
            self.mutex.acquire()
            obs_msg = copy.deepcopy(self.latest_observation_msg) 
            self.latest_observation_msg = None
            self.mutex.release()

            # initialize filter if state is None and save the current stamp as last_time
            if self.mu is None and obs_msg is not None:
                self.initialize_mu_and_sigma(obs_msg)
                last_time = rospy.Time.now()
                continue

            # do nothing until we have initialized the filter
            if last_time is None:
                continue

            # compute elapsed time from last prediction
            current_time = rospy.Time.now()
            delta_t = (current_time - last_time).to_sec()
            assert delta_t >= 0, "Negative delta_t = {}?".format(delta_t) # sanity check!

            # assemble A matrix: helps generate new state from prior state and elapsed time
            self.assemble_A_matrix(delta_t)

            # prediction step: predict new mean and covariance
            self.mu, self.Sigma = KF_predict_step(self.mu, self.Sigma, self.A, self.R)

            # save the time of when we made the prediction
            last_time = current_time

            # make more future predictions? (only for CPSC 559)
            tmp_mu = copy.deepcopy(self.mu)
            tmp_sigma = copy.deepcopy(self.Sigma)
            self.future_predictions(tmp_mu, tmp_sigma, current_time)

            # don't correct the state if we don't have a new observation
            if obs_msg is None:

                # store tracked state for visualization purposes
                self.store_tracked_state(self.mu, current_time)

                # wait to approximate the desired fps
                rospy.sleep(1.0/frame_rate)

                continue

            # assemble observation vector
            z = self.assemble_observation_vector(obs_msg)

            # measurement update step: correct mean and covariance
            self.mu, self.Sigma = KF_measurement_update_step(self.mu, self.Sigma, z, self.C, self.Q)

            # store tracked state for visualization purposes
            self.store_tracked_state(self.mu, current_time)

            # store observation for visualization purposes
            self.store_tracked_obs(z)

            # wait to approximate the desired fps
            rospy.sleep(1.0/frame_rate)


    def obs_callback(self, msg):
        """
        Observation callback. Stores the observation in self.latest_observation_msg.
        :param msg: PointStamped message with (x,y) location observation from the image
        """
        self.mutex.acquire()
        self.latest_observation_msg = msg
        if self.frame_id is None:
            self.frame_id = self.latest_observation_msg.header.frame_id
        elif self.frame_id != self.latest_observation_msg.header.frame_id:
            rospy.logwarn("Did the frame of the observations changed? Check the data!")
        self.mutex.release()


    def store_tracked_state(self, mu, current_stamp):
        """
        Store tracked state and publish result
        :param mu: state (position and vel)
        :param current_stamp: stamp for when the prediction was made
        """

        # build list of tracked positions so that we can draw a motion track for the target
        if self.tracked_positions.full():
            self.tracked_positions.get_nowait()

        self.tracked_positions.put_nowait(mu[:2, 0].transpose())

        # published filtered state
        msg = PoseStamped()
        msg.header.stamp = current_stamp
        msg.pose.position.x = mu[0,0]
        msg.pose.position.y = mu[1,0]
        msg.pose.position.z = mu[2,0]
        self.pub_filtered.publish(msg)

    
    def publish_position_markers(self):
        """
        Publish position markers for the observations and tracked states
        """
        if self.frame_id is None:
            return # can't publish markers without knowing their frame

        def make_marker(queue_obj, identifier, r=1.0, g=1.0, b=1.0, line_thickness=0.2):
            """
            Helper function to create marker for a single track of positions
            :param queue_obj: queue object with position data
            :param identifier: marker id
            :param r: red intensity [0,1]
            :param g: green intensity [0,1]
            :param b: blue intensity [0,1]
            """
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.type = marker.LINE_STRIP
            marker.action = marker.ADD
            marker.scale.x = line_thickness
            marker.color.a = 1.0
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.points = list(queue_obj.queue)
            marker.ns = "filtering"
            marker.id = identifier

        # create marker array object to publish all the tracks at once
        ma = MarkerArray()

        # create visualization marker for observations -- they will appear white
        if not self.observed_positions.empty():
            marker_obs = make_marker(self.observed_positions, 0, 1.0, 1.0, 1.0)
            ma.markers.append(marker_obs)

        # create visualization marker for filtered (current) positions -- they will appear as thick green lines
        if not self.tracked_positions.empty():
            marker_fil = make_marker(self.tracked_positions, 1, 0.0, 1.0, 0.0, 0.5)
            ma.markers.append(marker_fil)

        # publish marker array
        self.pub_markers.publish(ma)


    def store_tracked_obs(self, z):

        # build list of observed positions so that we can compared observed vs. tracked positions
        if self.observed_positions.full():
            self.observed_positions.get_nowait()

        self.observed_positions.put_nowait(z.transpose())


    def assemble_A_matrix(self, delta_t):
        """
        Method that assembles the A matrix for the KF_predict_step
        :param delta_t: elapsed time (in seconds) since last prediction
        """
        # TODO. Remove the "pass" line below and set self.A based on the elapsed time delta_t
        pass


    def assemble_C_matrix(self):
        """
        Method that assembles the C matrix for the KF_measurement_step
        """
        # TODO. Remove the "pass" line below and set self.C such that self.C x self.mu returns the expected measurement
        pass


    def initialize_process_covariance(self):
        """
        Method that sets the process covariance R for the filter node
        """
        # TODO. Remove the "pass" line below and set self.R
        pass


    def initialize_measurement_covariance(self):
        """
        Method that sets the process covariance Q for the filter node
        """
        # TODO. Remove the "pass" line below and set self.Q
        pass


    def initialize_mu_and_sigma(self, obs_msg):
        """
        Method that initializes the state (sets self.mu and self.Sigma).
        :param obs_msg Observation message with the latest measured position for the target
        """
        # TODO. Remove the "pass" line below and set self.mu and self.Sigma to their initial values here.
        pass


    def assemble_observation_vector(self, obs_msg):
        """
        Build the observation vector as a numpy array with the data from the Observation message
        :param obs_msg: latest Observation message that has been received by the node
        :return: numpy array representing the observation vector
        """
        # TODO. Complete. Build the numpy array z such that it corresponds to the observed target location.
        return z

    
    def future_predictions(self, mu, sigma, current_stamp):
        """
        Store tracked state and publish result
        :param mu: state (position and vel)
        :param sigma: covariance
        :param current_stamp: stamp for when the last prediction was made
        """
        # TODO. Complete (for 559). Write prediction code and publish /future_target message
        pass


if __name__ == '__main__':

    try:
        node = KalmanFilterNode()
    except rospy.ROSInterruptException:
        rospy.logwarn("ROS Interrupt.")
        pass
