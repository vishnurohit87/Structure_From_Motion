#!/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import plotly.graph_objects as go
import gtsam
from gtsam import (Cal3_S2, GenericProjectionFactorCal3_S2, NonlinearFactorGraph, SfmTrack, noiseModel,
                PinholeCameraCal3_S2, Point2, Point3, Pose3, PriorFactorPoint3, PriorFactorPose3, Values)

class sfm_helpers:
    def __init__(self, path, viz=False):
        self.path = path
        self.K = None
        self.viz = viz
    
    def getImages(self):    
        '''
        Reads and stores images in an array
        '''
        print(f"Loading images from {self.path}....")
        ims = [cv2.imread(os.path.join(self.path,file)) for file in sorted(os.listdir(self.path))]
        self.height, self.width, _ = ims[0].shape
        self.K = np.array([[1500, 0, self.width/2], [0, 1500, self.height/2], [0, 0, 1]])
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        images = [clahe.apply(cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))) for img in ims]
        print(f"Images loaded successfully!")
        
        if self.viz:
            fig, axs = plt.subplots(1, len(images), figsize=(24, 8))
            for i in range(len(images)):
                axs[i].imshow(images[i])  # Convert BGR to RGB
                axs[i].set_title(f'Image {i+1}')
                axs[i].axis('off')  # Turn off axis labels
        return images
    
    def findAndMatchFeatures(self, image1, image2, features):
        sift = cv2.SIFT_create(nfeatures=features, nOctaveLayers=3, contrastThreshold=0.04)
        f1, d1 = sift.detectAndCompute(image1, None)
        nms_mask, f1, d1 = self.non_maximum_suppression(f1, d1)

        f2, d2 = sift.detectAndCompute(image2, None)
        nms_mask, f2, d2 = self.non_maximum_suppression(f2, d2)

        # Find matches
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(d1, d2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        src_pts = np.array([f1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.array([f2[m.trainIdx].pt for m in good_matches])

        if self.viz:
            im1 = image1
            im2 = image2
            matching_result = cv2.drawMatches(im1, f1, im2, f2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.figure(figsize=(10, 5))
            plt.title('Feature Matching Result')
            plt.imshow(cv2.cvtColor(matching_result, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        return src_pts, dst_pts, good_matches
    
    def non_maximum_suppression(self, f, d):
        binary_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        responses = np.array([point.response for point in f])
        mask = np.flip(np.argsort(responses))
        
        point_list = np.rint([point.pt for point in f])[mask].astype(int)
        
        nms_mask = []
        for point, index in zip(point_list, mask):
            if binary_mask[point[1], point[0]] == 0:
                nms_mask.append(index)
                cv2.circle(binary_mask, tuple(point), 2, 255, -1)
        
        f = np.array(f)[nms_mask]
        d = np.array(d)[nms_mask]
                
        return nms_mask, f, d
   
    def essentialMat(self, src_pts, dst_pts):
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, cv2.RANSAC, 0.999, 1.0)
        inlier_src_pts = src_pts[mask.ravel() == 1]
        inlier_dst_pts = dst_pts[mask.ravel() == 1]

        return E, inlier_src_pts, inlier_dst_pts
    
    def posesFromE(self, E, img1_pts, img2_pts):   # For some reason this works better than recoverPose so kept this

        U, _, Vt = np.linalg.svd(E)

        Y = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R1 = U @ Y @ Vt
        R2 = U @ Y.T @ Vt

        t1 = U[:, 2]
        t2 = -1*U[:, 2]

        Rt_list = [(R1,t1), (R1,t2), (R2,t1), (R2,t2)]

        for i, (R, t) in enumerate(Rt_list):
            if np.linalg.det(R) < 0:
                Rt_list[i] = (-1*R, -1*t)

        # print(f"img1_pts: {img1_pts.shape}\nimg2_pts: {img2_pts.shape}\n")
        num_pos_pts = []
        for (R, t) in Rt_list:
            pos_pts = 0

            P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = self.K @ np.hstack((R, t.reshape(3, 1)))

            for pt1, pt2 in zip(img1_pts, img2_pts):
                # print(f"pt1: {pt1.shape}\npt2: {pt2.shape}\n")

                X = cv2.triangulatePoints(P1, P2, pt1, pt2)
                X /= X[3]
                X = X[0:3,:]

                chi1 = X[2]
                chi2 = R[2,:]@(X-np.reshape(t, X.shape))

                if(chi1>0 and chi2>0):
                    pos_pts+=1

            num_pos_pts.append(pos_pts)

        idx = np.argmax(num_pos_pts)
        (R, t) = Rt_list[idx]

        # Create transformation matrix from R and t
        Tr = np.vstack((np.hstack((R, t.reshape(3, 1))), np.array([0, 0, 0, 1])))

        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R, t.reshape(3, 1)))

        return R, t

    def triangulate_pts(self, pose_c1, pose_c2, filtered_src_pts, filtered_dst_pts):
        # Since cv2.triangulatePoints() takes in projection matrices in 3x4 form, we convert it to homogeneous
        pts_3d = cv2.triangulatePoints(pose_c1, pose_c2, filtered_src_pts.T, filtered_dst_pts.T) 

        print(f"pts_3d shape before masking = {pts_3d.shape}")

        # img1_pts = filtered_src_pts
        # img2_pts = filtered_dst_pts

        # Inhomoegnize points
        pts_3d = pts_3d/pts_3d[3,:]
        pts_3d = pts_3d[:3,:]

        # Select points with positive depth
        mask = pts_3d[2, :] > 0
        pts_3d = pts_3d[:, mask]
        img1_pts = filtered_src_pts[mask]
        img2_pts = filtered_dst_pts[mask]
        print(f"pts_3d shape after masking = {pts_3d.shape}")

        return pts_3d.T, img1_pts, img2_pts

    def posesFromPnP(self, pts_3d, pts_2d):
            success, rvec, t, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, self.K, None)

            # if success:
            R, _ = cv2.Rodrigues(rvec)

            # Create transformation matrix from R and t
            Tr = np.vstack((np.hstack((R, t.reshape(3, 1))), np.array([0, 0, 0, 1])))

            return Tr

class point_cloud:
    def __init__(self):
        self.tracks = {}
        self.camera_poses = [] 
        self.gtsam_camera_poses = []

    def addPoints(self, pts_3d, img1_pts, img2_pts, pose_idx1, pose_idx2):
        for P3d, img1_pt, img2_pt in zip(pts_3d, img1_pts, img2_pts):
            img1_pair = (pose_idx1, tuple(img1_pt))
            img2_pair = (pose_idx2, tuple(img2_pt))

            # Check if the 2D measurement and camera pose pairs exist in any track
            track_found = False
            for track_key in self.tracks:
                if img1_pair in self.tracks[track_key] or img2_pair in self.tracks[track_key]:
                    # print("Track found. Printing current and existing 3d point for comparison")
                    # print(tuple(P3d))
                    # print(track_key)
                    if img1_pair not in self.tracks[track_key]:
                        self.tracks[track_key].append(img1_pair)
                    if img2_pair not in self.tracks[track_key]:
                        self.tracks[track_key].append(img2_pair)
                    track_found = True
                    break

            # If no existing track found, create a new one
            if not track_found:
                self.tracks[tuple(P3d)] = [img1_pair, img2_pair]

        print(f"Number of tracks: {len(self.tracks)}")
        return self.tracks
        
    def common_pts(self, img1_pts, img2_pts):
        '''
        Returns lists of 3d points and their corresponding
        2d points in image2 for PnP
        '''
        match_3d = []
        match_2d = []
    
        for i, point in enumerate(img1_pts):
            for track, measurement in self.tracks.items():
                # print("TRACK1", track)
                if tuple(point) in [measurement[j][1] for j in range(len(measurement))]:
                    match_3d.append(np.array(track))              # Possible error 
                    match_2d.append(np.array(img2_pts[i]))

        return np.array(match_3d), np.array(match_2d)
    
    def addPoints_tracks(self, pts_3d, img1_pts, img2_pts, pose_idx1, pose_idx2):
        self.tracks = []
        for P3d, img1_pt, img2_pt in zip(pts_3d, img1_pts, img2_pts):
            p = Point3(np.array(P3d))
            track = SfmTrack(p)
            trackPoints = [tuple(t.p) for t in self.tracks]
            if tuple(track.p) not in trackPoints:                                        # Possible error in result since triangulated 3d point can be different
                track.addMeasurement(pose_idx1, Point2(np.array(img1_pt)))
                # print("Added new track")
                track.addMeasurement(pose_idx2, Point2(np.array(img2_pt)))
                # print("Added new track")
                self.tracks.append(track)
                # print("=======================================",tuple(track.p))
                # print(track.measurementMatrix())
                # print(track.measurement(0))
                # print("Added new track=============================================")
            
            else:
                track = self.tracks[trackPoints.index(tuple(track.p))]
                print("Track already exists", tuple(track.p))
                # print(track.measurementMatrix())
                # print(tuple(track.measurement(1)))
                m1 = (pose_idx1, tuple(img1_pt))
                m2 = (pose_idx2, tuple(img2_pt))
                im_pairs = []
                for i in  range(len(track.measurementMatrix())):
                    im_pairs.append((track.indexVector()[i], tuple(track.measurementMatrix()[i])))
                    print("IM PAIR: ", im_pairs[i])
                if m1 not in im_pairs:
                    track.addMeasurement(pose_idx1, Point2(np.array(img1_pt)))
                    print("Added measurement to existing track")
                    print(track.measurementMatrix())
                if m2 not in im_pairs:
                    track.addMeasurement(pose_idx2, Point2(np.array(img2_pt)))
                    print("Added measurement to existing track")
                    print(track.measurementMatrix())
        print(f"Number of tracks: {len(self.tracks)}")
            
        #for t in self.tracks:
        #    print(f"track.measurements = {[tuple(t.measurement(i)) for i in range(len(t.measurements))]}")

        return self.tracks

    def common_pts_tracks(self, img1_pts, img2_pts):
        '''
        Returns lists of 3d points and their corresponding
        2d points in image2 for PnP
        '''
        match_3d = []
        match_2d = []
    
        for i, point in enumerate(img1_pts):
            for track in self.tracks:
                if Point2(np.array(point)) in track.measurementMatrix():
                    match_3d.append(np.array(track.p))              # Possible error 
                    match_2d.append(np.array(img2_pts[i]))

        return np.array(match_3d), np.array(match_2d)

    def filter_points(self, perc):
        '''
        Filter top given percentile 3D points based on depth
        '''
        points_3d = np.array(list(self.tracks.keys()))
        depths = points_3d[:, 2]
        percentile = np.percentile(depths, perc)
        filtered_points_3d = points_3d[depths > percentile]

        for point in filtered_points_3d:
            del self.tracks[tuple(point)]
   
class plotter:
    def __init__(self, points_3d, poses, filter):
        self.points_3d = np.array(points_3d)
        self.poses = poses
        self.filter = filter

        # Filter out points with any coordinate beyond 10,000
        filtered_points_3d = self.points_3d[(np.abs(self.points_3d) <= filter).all(axis=1)]
        self.points_3d = filtered_points_3d
        
    def plot_3d_go(self):
        '''
        Uses Plotly to plot the 3D points and camera poses in a 3D plot
        '''
        total_colors = np.zeros((1, 3))
        x = self.points_3d[:, 0]
        y = self.points_3d[:, 1]
        z = self.points_3d[:, 2]

        fig = go.Figure()

        scatter = fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=1.2, color=total_colors/255)))

        # Plot camera poses
        for i in range(len(self.poses)):
            pose = self.poses[i]
            x, y, z = pose[0:3, 3]
            u, v, w = pose[0:3, 0:3] @ np.array([0, 0, 1])
            fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(size=3, color='red')))

        fig.show()

    def plot_3d_plt(self):
        '''
        Uses Matplotlib to plot the 3D points and camera poses in a 3D plot
        '''
        def plot_camera(ax, R, t, scale=1):
            # Define the camera pyramid vertices in camera coordinate system
            camera_body = np.array([
                [0, 0, 0],  # Camera center
                [1, 1, 2],  # Top-right corner
                [1, -1, 2],  # Bottom-right corner
                [-1, -1, 2],  # Bottom-left corner
                [-1, 1, 2]  # Top-left corner
            ]) * scale
            
            # Transform camera body to world coordinate system
            camera_body = (R @ camera_body.T).T + t
            
            # Define the six faces of the camera pyramid
            verts = [
                [camera_body[0], camera_body[1], camera_body[2]],  # Side 1
                [camera_body[0], camera_body[2], camera_body[3]],  # Side 2
                [camera_body[0], camera_body[3], camera_body[4]],  # Side 3
                [camera_body[0], camera_body[4], camera_body[1]],  # Side 4
                [camera_body[1], camera_body[2], camera_body[3], camera_body[4]]  # Base
            ]

            ax.add_collection3d(Poly3DCollection(verts, color='brown', alpha=0.5))

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D points
        ax.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], c='black', s=1)

        # Plot camera poses
        for pose in self.poses:
            # Extract the rotation and translation from the pose
            R = pose[:3, :3]
            t = pose[:3, 3]
            
            # Plot the camera with orientation
            plot_camera(ax, R, t)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def plot_3d_o3d(self, point_size=2, save=None):
        '''
        Uses Open3D to plot the 3D points and camera poses in a 3D plot
        '''

        # Create an Open3D point cloud object and negate y and z coordinates since image and world coordinates are different
        negated_points_3d = self.points_3d.copy()
        negated_points_3d[:, 1] = -negated_points_3d[:, 1]
        negated_points_3d[:, 2] = -negated_points_3d[:, 2]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(negated_points_3d)
        point_cloud.paint_uniform_color([0, 0, 0])  # Color all points black

        # Create a list to store camera spheres
        camera_spheres = []

        # Add camera poses as spheres and negate y and z coordinates
        for pose in self.poses:
            # Extract the translation from the pose
            t = pose[:3, 3]

            # Negate y and z coordinates in the translation vector
            t_negated = t.copy()
            t_negated[1] = -t_negated[1]
            t_negated[2] = -t_negated[2]

            # Create a sphere to represent the camera position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
            sphere.translate(t_negated)
            sphere.paint_uniform_color([1, 0, 0])  # Color the spheres red
            camera_spheres.append(sphere)

        # Create a visualizer and add geometry
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)
        for camera_sphere in camera_spheres:
            vis.add_geometry(camera_sphere)

        # Adjust the point size in the rendering options
        render_option = vis.get_render_option()
        render_option.point_size = point_size

        if save is not None:
            # Save the point cloud to a file
            o3d.io.write_point_cloud(save, point_cloud)
            print(f"Point cloud saved as {save}")

        # Run the visualizer
        vis.run()
        vis.destroy_window()

class gtsam_optimizer:
    def __init__(self, pc, K):
        self.pc = pc
        self.K = K

        # Create the factor graph
        self.graph = NonlinearFactorGraph()

        # Intialize the estimates
        self.initial_estimate = Values()

    def initialize_factor_graph(self):
        L = gtsam.symbol_shorthand.L
        X = gtsam.symbol_shorthand.X

        # extract relevant values from K to construct gtsam camera calibration parameters
        gtsam_K = Cal3_S2(self.K[0, 0], self.K[1, 1], 0.0, self.K[0, 2], self.K[1, 2])

        # Define the camera observation noise model
        measurement_noise = noiseModel.Isotropic.Sigma(2, 10.0)  

        # Add a prior on pose x1. This indirectly specifies where the origin is.
        # 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
        pose_noise = noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        factor = PriorFactorPose3(X(0), Pose3(self.pc.gtsam_camera_poses[0]), pose_noise)
        self.graph.push_back(factor)

        i=0
        for point, measurements in self.pc.tracks.items():
            for j, measurement in enumerate([measurements[k][1] for k in range(len(measurements))]):
                # if len(measuremens)>2:
                #     print(f"Point {i} has more thtan 2 measurements")
                factor = GenericProjectionFactorCal3_S2(Point2(np.array(measurement)), measurement_noise, X(measurements[j][0]), L(i), gtsam_K)
                self.graph.push_back(factor)
            i+=1

        print(f"No. of factors = {i}")
        # Add the prior on the position of the first landmarkurement = images[image_index].keypoints['sift'][keypoint_ind
        point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1)
        factor = PriorFactorPoint3(L(0), list(self.pc.tracks.keys())[0] , point_noise)
        self.graph.push_back(factor)  
        # graph.print('Factor Graph:\n')


        # initial_estimate.insert(K_key, K)
        for i, pose in enumerate(self.pc.gtsam_camera_poses):
            self.initial_estimate.insert(X(i), Pose3(pose))

        j = 0
        for point, measurements in self.pc.tracks.items():
            self.initial_estimate.insert(L(j), Point3(np.array(point)))
            j+=1

        return self.graph, self.initial_estimate, L, X
    
    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, params)
        result = optimizer.optimize()
        
        return result

def main():
    sfmh = sfm_helpers("buddha_images", False)
    images = sfmh.getImages()
    
    '''
    INITIAL TRIANGULATION
    '''
    pc = point_cloud()
    pc.camera_poses.append(np.eye(4)) # First camera pose is origin
    pc.gtsam_camera_poses.append(np.linalg.inv(pc.camera_poses[0]))

    print(f"Number of images = {len(images)}")
    print("\nStarting initial triangulation....")
    for i in range(len(images)-1):
        print(f"\nidx = {i}")
        if i==0:
            src_pts, dst_pts, good_matches = sfmh.findAndMatchFeatures(images[i], images[i+1], 1000)
        else:
            src_pts, dst_pts, good_matches = sfmh.findAndMatchFeatures(images[i], images[i+1], 5000)
        print(f"Number of good matched between images {i} and {i+1} = {len(good_matches)}")
        E, src_pts, dst_pts = sfmh.essentialMat(src_pts, dst_pts)

        if i==0:
            R, t = sfmh.posesFromE(E, src_pts, dst_pts)
            TrMat = np.vstack((np.hstack((R, t.reshape(3,1))), np.array([0, 0, 0, 1])))
            pc.camera_poses.append(TrMat)
            pc.gtsam_camera_poses.append(np.linalg.inv(TrMat))

            P1 = np.array(sfmh.K @ pc.camera_poses[i][:3,:])
            P2 = np.array(sfmh.K @ pc.camera_poses[i+1][:3,:])

            pts_3d, src_pts, dst_pts = sfmh.triangulate_pts(P1, P2, src_pts, dst_pts)
            
            pc.addPoints(pts_3d, src_pts, dst_pts, i, i+1)

        else:
            common_3d, common_2d_dst = pc.common_pts(src_pts, dst_pts)
            print(f"Number of 3D and 2D common points: {common_3d.shape}, {common_2d_dst.shape}")

            TrMat = sfmh.posesFromPnP(common_3d, common_2d_dst)
            pc.camera_poses.append(TrMat)
            pc.gtsam_camera_poses.append(np.linalg.inv(TrMat))

            P1 = sfmh.K @ pc.camera_poses[i][:3,:]
            P2 = sfmh.K @ pc.camera_poses[i+1][:3,:]

            pts_3d, src_pts, dst_pts = sfmh.triangulate_pts(P1, P2, src_pts, dst_pts)
            pc.addPoints(pts_3d, src_pts, dst_pts, i, i+1)

    # Plot Point Cloud
    pc.filter_points(90)
    pts_3d_plot = np.array([point for point in pc.tracks.keys()])
    print(f"Number of 3D points in the point cloud = {len(pts_3d_plot)}")
    
    pcPlot = plotter(pts_3d_plot, pc.gtsam_camera_poses, 4000)
    pcPlot.plot_3d_o3d(save="initial_point_cloud.ply")

    '''
    OPTIMIZATION USING GTSAM
    '''
    optimizer = gtsam_optimizer(pc, sfmh.K)

    #Initilize the factor graph
    graph, initial_estimate, L, X = optimizer.initialize_factor_graph()
    print('Initial Error = {}'.format(graph.error(initial_estimate)))

    # Optimize the graph and get the results
    result = optimizer.optimize()
    print('Final Error = {}'.format(graph.error(result)))

    # Plot Optimized point cloud
    opt_camera_poses = []
    optimized_pt_cloud = []

    i = 0
    for point in pc.tracks.keys():
        optimized_pt_cloud.append(result.atPoint3(L(i)))
        i+=1

    optimized_poses = []
    for i in range(len(images)):
        opt_camera_poses.append(result.atPose3(X(i)).matrix())
        R = opt_camera_poses[i][:3,:3]
        T = opt_camera_poses[i][:3,3]
        Tr = np.hstack((R, T.reshape(3, 1)))
        optimized_poses.append(Tr)

    print(len(optimized_poses))
    opt_pcPlot = plotter(np.array(optimized_pt_cloud), np.array(optimized_poses), 400)
    opt_pcPlot.plot_3d_o3d(save="optimized_point_cloud.ply")

if __name__ == "__main__":
    main()