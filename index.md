## IMU-Based Pedestrian Dead Reckoning with Learned Motion Model

We propose a tightly-coupled Extended Kalman Filter framework for IMU-only state estimation. We train a network that regresses 3D displacement estimates and its uncertainty, and tightly fuse the relative state measurement into a stochastic cloning EKF solving for pose, velocity and sensor biases. 

Here is a supplementary videos.

The blue trajectory is obtained with the proposed TLIO using only IMU data, while the green is obtained using a state-of-the art visual-inertial odometry algorithm.

[Staircase video](https://drive.google.com/open?id=1NIZilMaIGx05EUPfztoMxiR2g8P3C0TM)

More video will be disclosed on this website soon.


