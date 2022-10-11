# Assignment 3 (Extra questions for CPSC-559)

The sections below for Assignment 2 are only meant to be completed/answered by students taking CPSC-559. Students in CPSC-459 are welcome to try to solve the problems/tasks below and even come to office hours to discuss them. However, their answers to the questions in Parts V and VI of the Assignment 2 will not be considered during grading.

## Part V. Addressing Delayed Tracking

Once you have implemented your Kalman Filter (in Part IV of the assignment), you should
be able to visualize the robot following the target with the `follow_target.launch` script in the `shutter_kf` package. Unfortunately, though, if you speed up the motion of the target, you'll see the robot tracking it with a significant delay. For instance, if you run:

    ```bash
    $ roslaunch shutter_kf follow_target.launch add_noise:=true path_type:=circular fast_target:=true
    ```

    Then, you would see the green PoseStamped visualization lagging behind the true target, thus making the robot
    look a bit a way from the red ball, as shown below:

    <img src="docs/shutter_delay.png" width="480"/>


## Part VI. Non-Parametric Filtering

In this part of the assignment, you should think about how to implemennt a particle filter for your object tracking program (Part V). No code needs to be implemented, but answers must be included in your final report.

### Questions / Tasks

- **VI-1.** Would a particle filter be advantageous over a Kalman Filter for tracking your chosen object from Part V of this assignment? Explain in your report why or why not you expect a particle filter to be advantageous.

- **VI-2.** If you were to implement a particle filter for tracking your object in Part V, can you use the same transition (motion) model as in your Kalman Filter implementation? Do you need to modify it in any way? Explain in your report.

- **VI-3.** If you were to implement a particle filter for tracking your object in Part V, would it be beneficial to modify your measurement model from the one that you used for your Kalman Filter implementation? If yes, explain why and how you would modify your measurement model. If no, explain why as well in your report.
