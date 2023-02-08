import numpy as np
import gtsam
from gtsam.symbol_shorthand import B, V, X, L
import matplotlib.pyplot as plt
np.random.seed(0)

class InertialOdometryGraph(object):
    
    def __init__(self, IMU_PARAMS=None, BIAS_COVARIANCE=None):
        """
        Define factor graph parameters (e.g. noise, camera calibrations, etc) here
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.IMU_PARAMS = IMU_PARAMS
        self.BIAS_COVARIANCE = BIAS_COVARIANCE

    def add_imu_measurements(self, measured_poses, measured_acc, measured_omega, measured_vel, delta_t, n_skip, initial_poses=None):

        n_frames = measured_poses.shape[0]

        # Check if sizes are correct
        assert measured_poses.shape[0] == n_frames
        assert measured_acc.shape[0] == n_frames
        assert measured_vel.shape[0] == n_frames

        # Pose prior
        pose_key = X(0) # assigns a large random value (eg 8646911284551352320)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
        pose_0 = gtsam.Pose3(measured_poses[0])
        self.graph.push_back(gtsam.PriorFactorPose3(pose_key, pose_0, pose_noise))

        self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[0]))

        # IMU prior
        bias_key = B(0)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.5)
        self.graph.push_back(gtsam.PriorFactorConstantBias(bias_key, gtsam.imuBias.ConstantBias(), bias_noise))

        self.initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

        # Velocity prior
        velocity_key = V(0)
        velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, .5)
        velocity_0 = measured_vel[0]
        self.graph.push_back(gtsam.PriorFactorVector(velocity_key, velocity_0, velocity_noise))

        self.initial_estimate.insert(velocity_key, velocity_0)
        
        # Preintegrator
        accum = gtsam.PreintegratedImuMeasurements(self.IMU_PARAMS)

        # Add measurements to factor graph
        for i in range(1, n_frames):
            accum.integrateMeasurement(measured_acc[i], measured_omega[i], delta_t[i-1])
            if i % n_skip == 0:
                pose_key += 1
                DELTA = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.1 * np.random.randn()),
                                    gtsam.Point3(4 * np.random.randn(), 4 * np.random.randn(), 4 * np.random.randn()))
                '''
                `gtsam.Pose3` takes a rotation and a translation as input
                `gtsam.Rot3.Rodrigues` represents a rotation in 3D space using the Rodrigues' rotation formula. 
                 The three parameters passed in are angles of rotation around the x, y, and z axis respectively. 
                 In this case, the rotation angles are set to 0, and the rotation is generated randomly by multiplying 0.1 with
                 a random value from a normal distribution with mean of 0 and standard deviation of 1.
                `gtsam.Point3` similarly, generates translations randomly by multiplying 4 with a random value from a normal distribution
                '''
                self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[i]).compose(DELTA))
                '''
                `compose()` is a method of the gtsam.Pose3 class which composes two poses together, it returns a new composed Pose3 object
                Here it is used to add noise to the measured_pose (MAYBE)
                '''

                velocity_key += 1
                self.initial_estimate.insert(velocity_key, measured_vel[i])

                bias_key += 1
                self.graph.add(gtsam.BetweenFactorConstantBias(bias_key - 1, bias_key, gtsam.imuBias.ConstantBias(), self.BIAS_COVARIANCE))
                '''
                represents a measurement constraint between two variables with a constant bias. The constant bias term represents an additional
                error that is assumed to be constant for all measurements of this type. For example, if the measurement constraint is a 
                distance between two landmarks, the constant bias term could represent a systematic error in the sensor that measures that distance. 
                '''
                self.initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())
                '''The constant bias represents the steady-state errors that are present in the IMU measurements, and it is typically modeled 
                as a 3-dimensional vector for the angular velocity and a 3-dimensional vector for the linear acceleration. By modeling the constant
                bias as a separate variable in the gtsam graph, it allows for it to be estimated and corrected for, which improves the overall 
                accuracy of the IMU data'''

                # Add IMU Factor
                self.graph.add(gtsam.ImuFactor(pose_key - 1, velocity_key - 1, pose_key, velocity_key, bias_key, accum))

                # Reset preintegration
                accum.resetIntegration()

    def estimate(self, SOLVER_PARAMS=None):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, SOLVER_PARAMS)
        self.result = self.optimizer.optimize()
        return self.result
    
    def visualize(self, file_name):
      # write graph as dot file
      self.graph.saveGraph(file_name,self.initial_estimate)