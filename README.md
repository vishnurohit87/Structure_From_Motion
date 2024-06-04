# Structure from Motion (SfM)

This project implements a Structure from Motion (SfM) pipeline for 3D reconstruction from a series of 2D images. The pipeline includes an initial triangulation of 2D image points into 3D space using classical computer vision methods like feature extraction and matching, pose estimation, point cloud generation. The 3D point locations and poses are used as initial estimates to create a factor graph using GTSAM (Georgia Tech Smoothing and Mapping Library) and optimized using the Levenberg–Marquardt algorithm.

## Project Overview

The SfM pipeline involves the following steps:

1. **Image Loading and Preprocessing**: Load images and apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance features.
2. **Feature Detection and Matching**: Detect and match features between consecutive images using SIFT.
3. **Essential Matrix Computation**: Compute the essential matrix to find the relative pose between the first image pair.
4. **Pose Estimation**: Estimate camera poses with correct scale using Perspective-n-Point (PnP) pose computation for the consequent image pairs and triangulate 3D points.
5. **Point Cloud Generation**: Generate initial 3D point cloud and filter points based on depth.
6. **Optimization**: Optimize the point cloud and camera poses using GTSAM for minimizing covariances and improving accuracy.
7. **Plotting**: The code has 3 different functions in the plotter class to visualize the point cloud using different libraries. The current implementation uses Open3D which can be easily changed.

## Results
The images can be found in ./buddha_images folder. Reference image below:

<img src="buddha_images/buddha_006.png?raw=true" alt="BuddhaImage" width="300">


### Initial (un-optimized) point cloud with camera poses (red):
<img src="initial_pcl_with_camera_poses.png?raw=true" alt="Initial_pcl_with_cam_poses">
LEFT - Front View; RIGHT - Top View (for better visualization of camera poses)

### Optimized point cloud with camera poses (red).
<img src="optimized_pcl_with_camera_poses.png?raw=true" alt="Optimized_pcl_with_cam_poses">
LEFT - Front View;  RIGHT - Top View (for better visualization of camera poses)

## Requirements

The following packages are required to run the SfM pipeline:

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Open3D
- Plotly
- GTSAM

## Usage
Clone the repository, and run the following in the project folder:
```bash
python sfm.py
```

## File Structure

- `sfm.py`: Main script implementing the SfM pipeline.
- `buddha_images/`: Directory containing the input images.
- `initial_pcl_with_camera_poses.png`: Visualization of the initial point cloud with camera poses.
- `optimized_pcl_with_camera_poses.png`: Visualization of the optimized point cloud with camera poses.
- `initial_pointcloud.png`: Image of the initial point cloud.
- `optimized_point_cloud.png`: Image of the optimized point cloud.
- `StructureFromMotion.ipynb`: Jupyter Notebook implementation of the sfm.py code.
