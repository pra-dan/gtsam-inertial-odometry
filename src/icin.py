# python3 src/icin.py --n_skip 10 --n_frames 6000

import argparse
import time as tt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
import InertialOdometry as io
from utils import transform_from_rot_trans, get_theta, pose_from_gps

import gtsam
from gtsam.symbol_shorthand import B, V, X, L

plt.rc('font', size=16)

if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser(description='Inertial Odometry')
    parser.add_argument('--n_skip', dest='n_skip', type=int, default=1)
    parser.add_argument('--n_frames', dest='n_frames', type=int, default=None)
    args = parser.parse_args()

    fig, axs = plt.subplots(1, figsize=(12, 8), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.1, bottom=0.17)

    # write_dir = "tracker_outputs/"
    # print('==> Will write outputs to %s' % write_dir)
    # if not os.path.exists(write_dir):
    #   os.makedirs(write_dir)

    """ 
    Load data
    """
    data = pd.read_csv('icin.csv')
    # Number of frames
    if args.n_frames is None:
        n_frames = len(data.time)
    else:
        n_frames = args.n_frames

    # Time in seconds
    time = np.array([data.millis[k] - data.millis[0] for k in range(n_frames)])

    # Time step
    delta_t = np.diff(time)

    # Velocity
    measured_vel = np.array([[data.iloc[k].speed, 0.1, 0.1] for k in range(n_frames)])

    # Acceleration
    measured_acc = np.array([[data.iloc[k].ax, data.iloc[k].ay, data.iloc[k].az] for k in range(n_frames)])

    # Angular velocity 
    deg2rad = np.pi/360
    measured_omega = np.array([[deg2rad*data.iloc[k].rollrate, deg2rad*data.iloc[k].pitchrate, deg2rad*data.iloc[k].yawrate] for k in range(n_frames)])

    # Poses
    scale = None
    origin = None
    oxts_gt = []
    oxts = []
    noisy_input = []
    OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
    for i in range(len(data)):
        if scale is None: scale = np.cos(data.iloc[i].latitude *np.pi /180.)
        R, t = pose_from_gps(data.iloc[i],scale)
        if origin is None:origin = t
        TME = 50 #translation_max_error
        _ni = (t-origin)+[TME*np.random.uniform(-1,1) for _ in range(3)] # add a noise upto TME meters
        noisy_input.append(_ni)
        T_w_imu_gt = transform_from_rot_trans(R, t - origin)
        T_w_imu = transform_from_rot_trans(R, _ni)
        oxts.append(OxtsData(data.iloc[i], T_w_imu))
        oxts_gt.append(OxtsData(data.iloc[i], T_w_imu_gt))

    measured_poses = np.array([oxts[k][1] for k in range(n_frames)]) # accumulates all SE(3) mats (n,4,4)
    gt_poses = np.array([oxts_gt[k][1] for k in range(n_frames)])
    '''
    measured_poses: a 4x4 matrix used to represent the transformation from the world frame to the camera frame, which is composed by a rotation and a translation. 
    The rotation is represented by the upper left 3x3 sub-matrix, and the translation by the last column of the matrix.
    '''
    
    """
    GTSAM parameters
    """
    start=tt.time()
    print('==> Adding IMU factors to graph')

    g = 9.81

    # IMU preintegration parameters
    # Default Params for a Z-up navigation frame, such as ENU: gravity points along negative Z-axis
    IMU_PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
    I = np.eye(3)
    IMU_PARAMS.setAccelerometerCovariance(I * 0.2)
    IMU_PARAMS.setGyroscopeCovariance(I * 0.2)
    IMU_PARAMS.setIntegrationCovariance(I * 0.2)
    BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.4)

    """
    Solve IMU-only graph
    """
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
    params.setVerbosity('ERROR')
    params.setVerbosityLM('SUMMARY')

    print('==> Solving IMU-only graph')
    imu_only = io.InertialOdometryGraph(IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
    imu_only.add_imu_measurements(measured_poses, measured_acc, measured_omega, measured_vel, delta_t, args.n_skip)
    result_imu = imu_only.estimate(params)
    print(f'GTSAM: solved IMU-only graph in t={tt.time()-start} s')
    # save graph
    imu_only.visualize('io.dot')

    """
    Visualize results
    """
    print('==> Plotting results')
    x_gt = gt_poses[:,0,3]
    y_gt = gt_poses[:,1,3]
    
    theta_gt = np.array([get_theta(measured_poses[k,:3,:3])[2] for k in range(n_frames)])

    x_est_imu = np.array([result_imu.atPose3(X(k)).translation()[0] for k in range(n_frames//args.n_skip)]) 
    y_est_imu = np.array([result_imu.atPose3(X(k)).translation()[1] for k in range(n_frames//args.n_skip)]) 
    theta_est_imu = np.array([get_theta(result_imu.atPose3(X(k)).rotation().matrix())[2] for k in range(n_frames//args.n_skip)]) 

    noisy_input = np.array(noisy_input)
    axs.plot(noisy_input[:,0], noisy_input[:,1], color='b', label='noisy input')
    axs.plot(x_gt, y_gt, color='k', label='GT')
    axs.plot(x_est_imu, y_est_imu, color='r', label='IMU')
    axs.set_title('pre-integrate upto '+str(args.n_skip))
    axs.set_xlabel('$x\ (m)$')
    axs.set_ylabel('$y\ (m)$')
    axs.set_aspect('equal', 'box')
    plt.grid(True)

    plt.legend()
    plt.savefig('path.eps')

    # Plot pose as time series
    fig, axs = plt.subplots(3, figsize=(8, 8), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.15, bottom=0.17, hspace=0.5)

    # Plot x
    axs[0].grid(True)
    axs[0].plot(time, x_gt, color='k', label='GT')
    axs[0].plot(time[:n_frames-1:args.n_skip], x_est_imu, color='r', label='IMU')
    axs[0].set_xlabel('$t\ (s)$')
    axs[0].set_ylabel('$x\ (m)$')

    # Plot y
    axs[1].grid(True)
    axs[1].plot(time, y_gt, color='k', label='GT')
    axs[1].plot(time[:n_frames-1:args.n_skip], y_est_imu, color='r', label='IMU')
    axs[1].set_xlabel('$t\ (s)$')
    axs[1].set_ylabel('$y\ (m)$')

    # Plot theta
    axs[2].grid(True)
    axs[2].plot(time, theta_gt, color='k', label='GT')
    axs[2].plot(time[:n_frames-1:args.n_skip], theta_est_imu, color='r', label='IMU')
    axs[2].set_xlabel('$t\ (s)$')
    axs[2].set_ylabel('$\\theta\ (rad)$')
    plt.legend(loc='best')
    plt.savefig('poses.eps')

    # Plot pose as time series
    fig, axs = plt.subplots(3, figsize=(8, 8), facecolor='w', edgecolor='k')
    plt.subplots_adjust(right=0.95, left=0.15, bottom=0.17, hspace=0.5)
    # Plot x
    axs[0].grid(True)
    axs[0].plot(time[:n_frames-1:args.n_skip], np.abs(x_gt[:n_frames-1:args.n_skip] - x_est_imu), color='r', label='IMU')
    axs[0].set_xlabel('$t\ (s)$')
    axs[0].set_ylabel('$e_x\ (m)$')

    # Plot y
    axs[1].grid(True)
    axs[1].plot(time[:n_frames-1:args.n_skip], np.abs(y_gt[:n_frames-1:args.n_skip] - y_est_imu), color='r', label='IMU')
    axs[1].set_xlabel('$t\ (s)$')
    axs[1].set_ylabel('$e_y\ (m)$')

    # Plot theta
    axs[2].grid(True)
    axs[2].plot(time[:n_frames-1:args.n_skip], np.abs(theta_gt[:n_frames-1:args.n_skip] - theta_est_imu), color='r', label='IMU')
    axs[2].set_xlabel('$t\ (s)$')
    axs[2].set_ylabel('$e_{\\theta}\ (rad)$')
    
    plt.legend(loc='best')
    plt.savefig('errors.eps')
    plt.show()




