import rosbag
import numpy as np

import cv2
from cv_bridge import CvBridge

import sys
from tqdm import tqdm
from path import Path
import os

import pandas as pd

import matplotlib.pyplot as plt

from pose_syncing import sync_pose


class BagConverter(object):
    def __init__(self, bag_path, dst_folder):
        """
        Args: bag_path: Path to bag file
              dst_folder: Folder path to holds converted results 
        """
        self.dst_folder = Path(dst_folder)
        self.bag = rosbag.Bag(bag_path)
        self.bridge = CvBridge()
        self.full_ts = []
        self.valid_tstamps = []
        

    def save_intrinsics(self):
        intrinsic_generator = self.bag.read_messages(topics='/airsim_node/PX4/camera_1/Scene/camera_info')
        print("Reading Intrinsic Matrix......")
        for i, (topic, msg, t) in enumerate(intrinsic_generator):
            K_msg = msg.K # a tuple of (fx, 0, cx, 0, fy, cy, 0, 0, 1)
            K_matrix = np.array(K_msg).reshape([3,3]) # 3x3 intrinsic matrix
            np.savetxt(self.dst_folder / 'cam.txt', K_matrix)            
            break

    def save_rgb(self, save=False):
        # self.bag.read_messages()
        image_generator = self.bag.read_messages(topics='/airsim_node/PX4/camera_1/Scene')
        print("Reading Images  ......")
        
  
        for i, (topic, msg, t) in enumerate(tqdm(image_generator)):
            im = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            im_path = self.dst_folder / (str(i).zfill(6) + '.jpg')
            # Append every timestamps that has an image message published
            self.valid_tstamps.append(t.to_sec())
            # print(type(t.to_nsec()))
            if save:
                cv2.imwrite(im_path, im)       
    

        
    def save_position(self, to_map_frame = True):
        pose_generator = self.bag.read_messages('/mavros/local_position/odom')
        print("Reading Positions  ......")
        full_odom = []
        odom_ts = []
        first_pose = None

        for i, (topic, msg, t) in enumerate(tqdm(pose_generator)):
            # Append every timestamps that has an image message published
            px = float(msg.pose.pose.position.x) # float
            py = float(msg.pose.pose.position.y)
            pz = float(msg.pose.pose.position.z)

            pose_row = np.array([px, py, pz])
            full_odom.append(np.reshape(pose_row, [1,3]))
            odom_ts.append(t.to_sec())

        self.full_ts = np.array(odom_ts)
        # Array with full number Gt Poses before syncing with images timestamps, save it for later comparison
        full_odom = np.concatenate(full_odom, axis=0)  # [N , 3]

        assert len(full_odom) == len(self.full_ts)
        out_df = pd.DataFrame(np.concatenate([self.full_ts[:, np.newaxis], full_odom], axis=1),
                            columns=['timestamps', 'tx', 'ty', 'tz'])

        out_df.to_csv(self.dst_folder / 'raw_position.txt', sep=' ', index=False)

        print("syncing poses ..... ")
        # New Array with  GT Poses after syncing with images timestamps (Interpolated with neighbor GT poses)
        final_poses = sync_pose(valid_timestamps=np.array(self.valid_tstamps), odom_timestamps=self.full_ts, odom=full_odom) # [M, 7]
        first_pose = final_poses[0, :]
        print(first_pose)
        assert len(final_poses) == len(self.valid_tstamps)
        print(final_poses)
        out_df = pd.DataFrame(np.concatenate([np.array(self.valid_tstamps)[:, np.newaxis],final_poses], axis=1),
                            columns=['timestamps', 'tx', 'ty', 'tz'])

        out_df.to_csv(self.dst_folder / 'interpolated_postition.txt', sep=' ', index=False)
        if to_map_frame:
            out_df = pd.DataFrame(np.concatenate([np.array(self.valid_tstamps)[:, np.newaxis],final_poses-first_pose], axis=1),
                            columns=['timestamps', 'tx', 'ty', 'tz'])
            out_df.to_csv(self.dst_folder / 'positionconverted_to_map_frame.txt', sep=' ', index=False)
        
    #-0.001747999664 -0.002930306695 0.000226489639
    def quat2mat(self, w, x, y, z):
        """
        Args:  w,x,y,z: 4 quarternion coefficients
        Return: Corresponing 3x3 Rotation matrix 
        https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        """
        ww, wx, wy, wz = w*w, w*x, w*y, w*z
        xx, xy, xz = x*x, x*y, x*z
        yy, yz = y*y, y*z
        zz = z*z

        n = ww + xx + yy + zz

        s = 0 if n < 1e-8 else 2 / n
        
        R = np.array([1 - s*(yy+zz),  s*(xy-wz)   ,  s*(xz+wy), 
                      s*(xy+wz)    ,  1 - s*(xx+zz), s*(yz-wx),
                      s*(xz-wy),      s*(yz+wx),     1-s*(xx+yy)]).reshape([3,3])

        return R

    def save_pose(self):
        pose_generator = self.bag.read_messages('/mavros/local_position/odom')
        print("Reading Poses  ......")
        full_odom = []
        odom_ts = []
       
        for i, (topic, msg, t) in enumerate(tqdm(pose_generator)):
            px = float(msg.pose.pose.position.x) # float
            py = float(msg.pose.pose.position.y)
            pz = float(msg.pose.pose.position.z)

            ox = float(msg.pose.pose.orientation.x) # float
            oy = float(msg.pose.pose.orientation.y)
            oz = float(msg.pose.pose.orientation.z)
            ow = float(msg.pose.pose.orientation.w)

            t_vec = np.array([px,py,pz]).reshape([3,1]) # 3x1 Translational vector
            rot_mat = self.quat2mat(ow, ox, oy, oz) # 3x3 Rotation matrix
            T_mat = np.concatenate([rot_mat, t_vec], axis=1) # 3x4 Transformation matrix
            # Append the flattened Transformation (Pose vector) and the corresponding timestamps
            full_odom.append(T_mat.reshape([-1,12])) 
            odom_ts.append(t.to_sec())
            
        # Array with full number Gt Poses before syncing with images timestamps, save it for later comparison
        full_odom = np.concatenate(full_odom, axis=0)  # [N , 12]
        np.savetxt(self.dst_folder / 'world_poses_full.txt', full_odom)
        
        # Array of every timestamps that a Gt odometry message is published
        odom_ts = np.array(odom_ts) # [N,]
            
        print("syncing poses ..... ")
        # New Array with  GT Poses after syncing with images timestamps (Interpolated with neighbor GT poses)
        final_poses = sync_pose(valid_timestamps=np.array(self.valid_tstamps), odom_timestamps=odom_ts, odom=full_odom) # [M, 12]
        np.savetxt(self.dst_folder / 'world_poses.txt', final_poses)

    def save_gps(self):
        pose_generator = self.bag.read_messages('/airsim_node/PX4/gps/Gps')
        print("Reading GPS  ......")
        full_odom = []
        odom_ts = []
       
        for i, (topic, msg, t) in enumerate(tqdm(pose_generator)):
            lat = float(msg.latitude)
            long = float(msg.longitude) # float
            alt = float(msg.altitude) # float

            t_vec = np.array([lat, long, alt]).reshape([1,3]) # 3x1 Translational vector
            full_odom.append(t_vec)
            odom_ts.append(t.to_sec())

        # Array with full number Gt Poses before syncing with images timestamps, save it for later comparison
        full_odom = np.concatenate(full_odom, axis=0)  # [N , 3]
        np.savetxt(self.dst_folder / 'gps_full.txt', full_odom)
        
        # Array of every timestamps that a Gt odometry message is published
        odom_ts = np.array(odom_ts) # [N,]
            
        print("syncing GPS ..... ")
        # New Array with  GT Poses after syncing with images timestamps (Interpolated with neighbor GT poses)
        final_poses = sync_pose(valid_timestamps=np.array(self.valid_tstamps), odom_timestamps=odom_ts, odom=full_odom) # [M, 12]
        # np.savetxt(self.dst_folder / 'gps_interpolated.txt', final_poses)
        out_df = pd.DataFrame(np.concatenate([np.array(self.valid_tstamps)[:, np.newaxis],final_poses], axis=1),
                            columns=['timestamps', 'lat', 'long', 'alt'])

        out_df.to_csv(self.dst_folder / 'gps_converted.txt', sep=' ', index=False)



    def worldpose2odom(self):
        """
        Convert to odometry (relative poses w.r.t first pose of the trajectory)
        """
        world_poses = np.loadtxt(self.dst_folder / 'world_poses.txt') # [seq_length, 12]
        world_poses = world_poses.reshape([-1, 3, 4]) # [seq_length, 3, 4]
        first_pose = world_poses[:1, :, :] # [1, 3, 4]
        first_rot_matrices = first_pose[:, :, :3] # [1, 3, 3] 
        first_t_vec = first_pose[:, :, -1:] # [1, 3, 1]
        rot_matrices_inv = np.linalg.inv(first_rot_matrices)  # [1, 3, 3]
        t_vec_inv  = -rot_matrices_inv @ first_t_vec # [1, 3, 1]
        world2first = np.concatenate([rot_matrices_inv, t_vec_inv], axis=-1) # [1,3,4]
        
        odom = world2first[..., :3] @ world_poses # Multiple with rotation [seq_length, 3, 4]
        odom[... , -1:] += world2first[..., -1:] # Add translation [seq_length, 3, 4]
        
        np.savetxt(self.dst_folder / 'poses.txt', odom.reshape([-1, 12]))

    def save_depth(self):
        
        # image_generator = self.bag.read_messages(topics='/airsim_node/PX4/camera_1/DepthPlanar')
        image_generator = self.bag.read_messages(topics='/camera/depth/image_rect_raw')
        print("Reading Depth Images  ......")
     
        for i, (topic, msg, t) in enumerate(tqdm(image_generator)):

            h, w = msg.height, msg.width
            dtype = np.dtype("uint16") 
            # Depth message from airsim is encoded as 32FC1
            print(len(msg.data))
            im = np.ndarray(shape=(h, w),
                           dtype=dtype, buffer=msg.data)  
            out = np.copy(im)
            out.setflags(write=1)
            out = out / 1000.
            # Clip depth values that exceesds 100 (m)
            # out[im>=100] = 100
            # Save to binary file
            depth_path = self.dst_folder / (str(i).zfill(6) + '.npy')
            # np.save(depth_path, out)

            if i == 550:
                plt.imshow(out, 'gray')
                plt.show()
                print(out.max())
                break



class BagDataReader(object):
    def __init__(self, raw_folder: str, 
                       tgt_folder: str,
                       get_depth: True,
                       get_pose: True
                       ):
        """
        Args: raw_folder: Path to folder containing bag files for each sequences
              tgt_folder: Path to destination folder, which hosts results subfolders arranged as
                    data_0
                           |0000000.jpg
                           |0000000.npy (if get_depth is True)
                           |0000001.jpg
                           |0000001.npy
                           |...
                           |cam.txt
                           |poses.txt (if get_pose is True)
                    data_1
                    data_2
                    .....
        """
        self.raw_folder = Path(raw_folder)
        self.tgt_folder = Path(tgt_folder)
        self.scene_names = [n[:-4] for n in os.listdir(raw_folder)] 
        self.get_depth = get_depth
        self.get_pose = get_pose

    def collect_single_scene(self, scene_name):
        """
        Args: Scene name in scene list
        """
        bag_path = self.raw_folder / (scene_name + '.bag')
        dst_folder = self.tgt_folder / scene_name
        if os.path.exists(dst_folder):
            os.mkdir(dst_folder)

        converter = BagConverter(bag_path, dst_folder)
        converter.save_intrinsics()
        converter.save_rgb()
        if self.get_pose:
            converter.save_pose()
        if self.get_depth:
            converter.save_depth()

    def read_multiple_scenes(self):
        for scene_name in self.scene_names:
            self.collect_single_scene(scene_name)


        

def main(args):

    converter = BagConverter('/home/minh/duy/subset_datasheet_2.bag', '/home/minh/tien/data-with-gps')
    # converter.save_intrinsics()
    converter.save_rgb(save=True)
    # converter.save_pose()
    converter.save_gps()
    converter.save_position(to_map_frame=True)
    # converter.save_depth()
    # converter.worldpose2odom()

if __name__ == '__main__':
    main(sys.argv)
