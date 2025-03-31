# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time
import math
import numpy as np
import os
import torch
from isaacgym import gymutil
import math
from perlin_noise import PerlinNoise
import random
import matplotlib.pyplot as plt
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from scipy import interpolate

from isaacgymenvs.tasks.base.vec_task import VecTask

from typing import Tuple, Dict
from isaacgym import gymutil

torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)


DEBUG = True

class Gogoro(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        
        
        # Create helper geometry used for visualization
        # Create an wireframe axis
        self.axes_geom = gymutil.AxesGeometry(0.1)
        # Create an wireframe sphere
        self.sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        self.sphere_pose = gymapi.Transform(r=self.sphere_rot)
        self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, self.sphere_pose, color=(1, 0, 0))

        self.viewer = virtual_screen_capture
        
        
        
        self.cfg = cfg
        
        
        #Driving parameters
        self.tilt_limit  = torch.tensor(np.radians(self.cfg["env"]["tiltLimit"]))
        self.min_speed   = torch.tensor(self.cfg["env"]["speedRange"][0])
        self.max_speed   = torch.tensor(self.cfg["env"]["speedRange"][1])
        self.range_speed = torch.tensor(self.max_speed - self.min_speed)
        self.set_speed   = torch.tensor(self.cfg["env"]["setSpeed"])
        self.min_delta_yaw = torch.tensor(np.radians(self.cfg["env"]["yawRange"][0]))
        self.max_delta_yaw = torch.tensor(np.radians(self.cfg["env"]["yawRange"][1]))



        #Targets
        self.set_target_delta_yaw = self.cfg["env"]["setDeltaYaw"]
        if self.set_target_delta_yaw != None:
            self.set_target_delta_yaw = np.radians(self.set_target_delta_yaw)

        #Sim Params
        self.noise_scale = self.cfg["env"]["noise_scale"]
        self.n_envs = self.cfg["env"]["numEnvs"]
        self.max_episode_length = torch.tensor(self.cfg["env"]["maxEpisodeLength"])

        # Same as kafka arm env
        self.action1_scale = 1.0
        self.action0_scale = 1.0

        self.root_velocity_scale = 0.25
        

        #Gym params
        num_obs = 26
        num_acts = 2
        
        self.cfg["env"]["numObservations"]  = num_obs
        self.cfg["env"]["numActions"]       = num_acts

        super().__init__(
                            config=self.cfg,
                            rl_device=rl_device,
                            sim_device=sim_device,
                            graphics_device_id=graphics_device_id,
                            headless=headless,
                            virtual_screen_capture=virtual_screen_capture,
                            force_render=force_render
                        )
        self.dt = self.sim_params.dt
        

        #LINKING THE TENSORS
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self._state_dof   = self.gym.acquire_dof_state_tensor(self.sim)
        self._rb_states   = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _jacobian         = self.gym.acquire_jacobian_tensor(self.sim, "Gogoro")


        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor) # PyTorch-wrapped tensor
        self.state_dof   = gymtorch.wrap_tensor(self._state_dof)
        self.rb_states = gymtorch.wrap_tensor(self._rb_states)
        # jacobian shape : num_envs, num_links, 6, num_dofs + 6
        self.jacobian = gymtorch.wrap_tensor(_jacobian)


        self.root_positions    = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_angular_vels = self.root_tensor[:, 10:13]
        self.dof_pos = self.state_dof.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.state_dof.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        self.num_rigidbody_per_actor = int(self.gym.get_sim_rigid_body_count(self.sim)/self.num_envs)
        self.links_state = self.rb_states.view(self.num_envs,self.num_rigidbody_per_actor, 13)


        # Set initial state to DOF targets
        for j_name in self.cfg["joints_pos"]:
            self.dof_pos[:, self.dof_name_to_id[j_name]] = self.cfg["joints_pos"][j_name]


        # l_leg_hip_y -> 0
        # l_leg_hip_r -> 1
        # l_leg_hip_p -> 2
        # l_leg_kn_p -> 3
        # l_leg_an_p -> 4
        # l_leg_an_r -> 5
        # r_leg_hip_y -> 6
        # r_leg_hip_r -> 7
        # r_leg_hip_p -> 8
        # r_leg_kn_p -> 9
        # r_leg_an_p -> 10
        # r_leg_an_r -> 11
        # rear_wheel_joint -> 12
        # steering_joint -> 13
        # front_wheel_joint -> 14
        # torso_y -> 15
        # head_y -> 16
        # head_p -> 17
        #================================actionables
        # l_arm_sh_p1 -> 18
        # l_arm_sh_r -> 19
        # l_arm_sh_p2 -> 20
        # l_arm_el_y -> 21
        # l_arm_wr_r -> 22
        # l_arm_wr_y -> 23
        # l_arm_wr_p -> 24
                            # l_arm_grip -> 25
                            # l_arm_grip_1 -> 26
        # r_arm_sh_p1 -> 27
        # r_arm_sh_r -> 28
        # r_arm_sh_p2 -> 29
        # r_arm_el_y -> 30
        # r_arm_wr_r -> 31
        # r_arm_wr_y -> 32
        # r_arm_wr_p -> 33
                            # r_arm_grip -> 34

        self.leftDOFIndex =np.array([18,19,20,21,22,23,24])
        self.rightDOFIndex=np.array([27,28,29,30,31,32,33])

        self.rgdb_to_index      = self.gym.get_actor_rigid_body_dict(self.envs[0],self.handles[0]) #its the same for all the envs anyway
        # back  ->  0
        # body  ->  1
        # cam_link  ->  2
        # chest_link  ->  3
        # front  ->  4
        # head  ->  5
        # head_p_link  ->  6
        # head_y_link  ->  7
        # imu_link  ->  8
        # l_arm_el_y_link  ->  9
                                            # l_arm_end_link  ->  10
        # l_arm_grip_1_link  ->  11
        # l_arm_grip_link  ->  12
        # l_arm_sh_p1_link  ->  13
        # l_arm_sh_p2_link  ->  14
        # l_arm_sh_r_link  ->  15
        # l_arm_wr_p_link  ->  16
        # l_arm_wr_r_link  ->  17
        # l_arm_wr_y_link  ->  18
        # l_foot_ft_link  ->  19
        # l_leg_an_p_link  ->  20
        # l_leg_an_r_link  ->  21
        # l_leg_foot_link  ->  22
        # l_leg_hip_p_link  ->  23
        # l_leg_hip_r_link  ->  24
        # l_leg_hip_y_link  ->  25
        # l_leg_kn_p_link  ->  26
                                            # l_steering_handle_end  ->  27
        # lidar_link  ->  28
        # pelvis_link  ->  29
        # r_arm_el_y_link  ->  30
                                            # r_arm_end_link  ->  31
        # r_arm_grip_1_link  ->  32
        # r_arm_grip_link  ->  33
        # r_arm_sh_p1_link  ->  34
        # r_arm_sh_p2_link  ->  35
        # r_arm_sh_r_link  ->  36
        # r_arm_wr_p_link  ->  37
        # r_arm_wr_r_link  ->  38
        # r_arm_wr_y_link  ->  39
        # r_foot_ft_link  ->  40
        # r_leg_an_p_link  ->  41
        # r_leg_an_r_link  ->  42
        # r_leg_foot_link  ->  43
        # r_leg_hip_p_link  ->  44
        # r_leg_hip_r_link  ->  45
        # r_leg_hip_y_link  ->  46
        # r_leg_kn_p_link  ->  47
                                            # r_steering_handle_end  ->  48
        # realsense_link  ->  49
        # steering_head_force_point  ->  50


        self.init_dof_pose = self.dof_pos.clone()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_reset_tensor = self.root_tensor.clone() # Copy root tensor
        self.reset_state_dof   = self.state_dof.clone()
        self.default_dof_pos   = self.dof_pos.clone()
        self.root_reset_tensor[:, 7:13] = 0 # Set all velocities to 0

        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.target_delta_yaw = torch.zeros_like(self.root_angular_vels[:, 2])

        # The noise scale added to the observations
        self.noise_scale_tensor = torch.ones(self.num_obs, device=self.device,
                dtype=torch.float32) * self.noise_scale

    
    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if(DEBUG):
            self._create_ground_plane()
        else:
            self._create_ground_realistic()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        #TODO: create random plane here
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.7
        plane_params.dynamic_friction = 0.7
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_ground_realistic(self):
        self.terrain = Terrain(num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices =  self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -20
        tm_params.transform.p.y = -250
        tm_params.transform.p.z = -0.1
        tm_params.static_friction = 0.7
        tm_params.dynamic_friction = 0.7
        tm_params.restitution = 0.0
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "/home/ugo/NVIDIA_OMNIVERSE/PROJECTS/IsaacGymEnvs/assets"
        asset_file = "urdf/gogoro/urdf/gogoro_and_thormang3.urdf"
            # Load asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = False
        asset_options.replace_cylinder_with_capsule = False
        asset_options.flip_visual_attachments = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        
        
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)


        # Get number of DOFs and DOF names
        self.num_dof   = self.gym.get_asset_dof_count(asset)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        # Helper dictionary to map joint names to tensor ID
        self.dof_name_to_id = {k: v for k, v in zip(self.dof_names, np.arange(self.num_dof))}

        dof_props = self.gym.get_asset_dof_properties(asset)
        # Everything set to pos so they all stay in-place
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][i] = 10000.0
            dof_props['damping'][i]   = 200.0

        # Set velocity mode on back wheel
        dof_props["driveMode"][self.dof_name_to_id['rear_wheel_joint']] = gymapi.DOF_MODE_VEL
        dof_props["stiffness"][self.dof_name_to_id['rear_wheel_joint']] = 1000.0
        dof_props["damping"][self.dof_name_to_id['rear_wheel_joint']] = 10.0 

        # Front wheel mode set to None so it runs freely
        dof_props["driveMode"][self.dof_name_to_id['front_wheel_joint']] = gymapi.DOF_MODE_NONE
        dof_props["stiffness"][self.dof_name_to_id['front_wheel_joint']] = 0.0
        dof_props["damping"][self.dof_name_to_id['front_wheel_joint']] = 0.0
        
        # Set position mode for steering joint
        dof_props["driveMode"][self.dof_name_to_id['steering_joint']] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][self.dof_name_to_id['steering_joint']] =  1000.0
        dof_props["damping"][self.dof_name_to_id['steering_joint']]   =  200

        self.steer_upper_limit = dof_props["upper"][13]
        self.steer_lower_limit = dof_props["lower"][13]
        self.steering_range = self.steer_upper_limit - self.steer_lower_limit


        # set up the env grid
        envs_per_row = num_per_row
        env_spacing  = spacing
        env_lower    = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper    = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # Set the target position of the DOFs according to the cfg
        self.pos_targets = np.zeros(self.num_dof).astype('f')
        # Set DOF initial position according to CFG
        for j_name in self.cfg["joints_pos"]:
            self.pos_targets[self.dof_name_to_id[j_name]] = self.cfg["joints_pos"][j_name]

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 1.1)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)


        self.envs = []
        self.handles = []
        # create and populate the environments
        for i in range(num_envs):
            ref_env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            actor_handle = self.gym.create_actor(ref_env, asset, start_pose, "Gogoro", i, 1)
            self.envs.append(ref_env)
            self.handles.append(actor_handle)
            self.gym.set_actor_dof_properties(ref_env, actor_handle, dof_props)
            self.gym.set_actor_dof_position_targets(ref_env, actor_handle, self.pos_targets)
            
        p = np.tile(self.pos_targets, self.num_envs)
        self.positions_tensor = torch.tensor(p, dtype=torch.float32, device=self.device)




    def pre_physics_step(self, actions):
        
        #create array conatining the velocities
        np_speeds = np.zeros((self.num_envs,self.num_dof))
        #setting the rear wheel vel to 1000
        np_speeds[:,self.dof_name_to_id['rear_wheel_joint']] = actions[:,1]*self.action1_scale #self.dof_name_to_id['rear_wheel_joint']
        #convert to tensor
        torch_speeds = torch.tensor(np_speeds.astype('f'),device=self.device)
        #nvidia bullshit
        targets = gymtorch.unwrap_tensor(torch_speeds)
        #calling the function
        self.gym.set_dof_velocity_target_tensor(self.sim,targets)
        
        left_handle_index  = self.rgdb_to_index["l_steering_handle_end"]
        left_hand_index    = self.rgdb_to_index["l_arm_end_link"]
        left_handle_pos = self.links_state[:,left_handle_index,0:3]
        left_hand_pos   = self.links_state[:,left_hand_index,0:3]
        left_hand_ori   = self.links_state[:,left_hand_index,3:7]  
        
        right_handle_index  = self.rgdb_to_index["r_steering_handle_end"]
        right_hand_index    = self.rgdb_to_index["r_arm_end_link"]
        right_handle_pos = self.links_state[:,right_handle_index,0:3]
        right_hand_pos   = self.links_state[:,right_hand_index,0:3]
        right_hand_ori   = self.links_state[:,right_hand_index,3:7]  
        
        j_eefR = self.jacobian[:, right_hand_index, :, self.rightDOFIndex+6 ]
        j_eefL = self.jacobian[:, left_hand_index, :, self.leftDOFIndex+6   ]
        
        goal_posR = right_handle_pos
        goal_rotR = right_hand_ori
        goal_posL = left_handle_pos
        goal_rotL = left_hand_ori

        Rpos_err = goal_posR - right_hand_pos
        Lpos_err = goal_posL - left_hand_pos

        Rorn_err = orientation_error(goal_rotR, right_hand_ori)
        Lorn_err = orientation_error(goal_rotL, left_hand_ori)
        
        Rdpose = torch.cat([Rpos_err, Rorn_err], -1).unsqueeze(-1)
        Ldpose = torch.cat([Lpos_err, Lorn_err], -1).unsqueeze(-1)

        # solve damped least squares
        j_eefR_T = torch.transpose(j_eefR, 1, 2)
        j_eefL_T = torch.transpose(j_eefL, 1, 2)

        lmbdaR = torch.eye(6, device=self.device) * (0.1 ** 2)
        lmbdaL = torch.eye(6, device=self.device) * (0.1 ** 2)
        
        uR = (j_eefR_T @ torch.inverse(j_eefR @ j_eefR_T + lmbdaR) @ Rdpose).view(self.n_envs, 7)
        uL = (j_eefL_T @ torch.inverse(j_eefL @ j_eefL_T + lmbdaL) @ Ldpose).view(self.n_envs, 7)

        test = self.dof_pos.clone()
        test[:,self.rightDOFIndex]+=uR
        test[:,self.leftDOFIndex] +=uL

        test[:,self.dof_name_to_id['steering_joint']]= actions[:,0]*self.action0_scale

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(test))


    
    def post_physics_step(self):

        self.progress_buf += 1
        # Check which environments should be reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Compute observations
        self.compute_observations()

        # Compute rewards
        self.compute_reward()


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_gogoro_reward(
                                                                    self.obs_buf,
                                                                    self.progress_buf,
                                                                    self.reset_buf,
                                                                    self.tilt_limit, 
                                                                    self.max_episode_length,
                                                                    #self.root_velocity_scale
                                                                    )

    def compute_observations(self):
        # Refresh the tensors
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        #self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)


        # Randomly generate offsets for the target delta yaw from a gaussian
        # distribution
        random_offsets = torch.rand_like(self.target_delta_yaw) * 0.2 - 0.1
        self.target_delta_yaw = torch.where(self.progress_buf % 150 == 0, 
                self.target_delta_yaw + random_offsets,
                self.target_delta_yaw)
        self.target_delta_yaw = torch.where(self.target_delta_yaw < self.min_delta_yaw,
                torch.ones_like(self.target_delta_yaw) * self.min_delta_yaw,
                self.target_delta_yaw)
        self.target_delta_yaw = torch.where(self.target_delta_yaw > self.max_delta_yaw,
                torch.ones_like(self.target_delta_yaw) * self.max_delta_yaw,
                self.target_delta_yaw)
        
        # Override target delta yaw tensor if value was set in config file
        if self.set_target_delta_yaw != None:
            self.target_delta_yaw = torch.ones_like(self.target_delta_yaw) * self.set_target_delta_yaw

        self.obs_buf[:] = compute_gogoro_observations( 
                                                        self.root_tensor,
                                                        self.dof_pos,
                                                        self.dof_vel,
                                                        self.target_delta_yaw,
                                                        self.inv_start_rot,
                                                        self.noise_scale_tensor,
                                                        self.max_speed,
                                                        self.links_state,
                                                        )

        
    def reset_idx(self, env_ids):
        # Reset environments indicated by `env_ids`
        #print("Reset called")
        # Set sim to reset tensor for envs in env_ids
        #TODO : add randomness here
        
        env_ids_int32  = env_ids.to(dtype=torch.int32)
        _env_ids_int32 = gymtorch.unwrap_tensor(env_ids_int32) 
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                    gymtorch.unwrap_tensor(self.root_reset_tensor),
                                                    _env_ids_int32, len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.reset_state_dof),
                                                _env_ids_int32, len(env_ids_int32))
        # Reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0



            
            
#####################################################################
###=========================jit functions=========================###
#####################################################################

#@torch.jit.script
def control_ik(damping, j_eef, num_envs,device,dpose):
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)



@torch.jit.script
def compute_gogoro_reward(obs_buf, progress_buf, reset_buf,tilt_limit, max_episode_length):#, delta_scale):
    # Reset if over iteration limit
    reset = torch.where(progress_buf > max_episode_length, 1, reset_buf)
    # Reset if over tilt limit
    reset = torch.where(torch.abs(obs_buf[:, 0]) > tilt_limit, 1, reset)

    dist_hands_to_handles = obs_buf[:,-1] +  obs_buf[:,-2]

    curr_delta_yaw = obs_buf[:, 3] #* delta_scale
    desired_delta = obs_buf[:, 6]
    diff_delta = curr_delta_yaw - desired_delta

    R1 = 1.0 - (torch.abs(diff_delta) ** 0.2)
    R2 = -1.0 * (dist_hands_to_handles*2)
    reward = R1 + R2
    
    #print("reward=",reward[0],"[ R1 =",R1[0],"| R2 =", R2[0],"]")

    # Negative reward for falling over
    reward = torch.where(torch.abs(obs_buf[:, 0]) > tilt_limit, torch.ones_like(reward) * -200.0, reward)
    return reward, reset





@torch.jit.script
def compute_rot_gogoro(torso_rotation, inv_start_rot, velocity, ang_vel):

    torso_quat       = quat_mul(torso_rotation,inv_start_rot)
    roll, pitch, yaw = get_euler_xyz(torso_quat)
    vel_loc       = quat_rotate_inverse(torso_quat,velocity)
    angvel_loc    = quat_rotate_inverse(torso_quat,ang_vel)


    return vel_loc, angvel_loc, roll, pitch, yaw



@torch.jit.script
def compute_gogoro_observations(root_state, dof_pos, dof_vel, desired_delta_yaw, inv_start_rot, noise_scale, max_rear_wheel_velocity,links_state):
    
    root_position     = root_state[:, 0:3]
    root_orientations = root_state[:, 3:7]
    root_linvel       = root_state[:, 7:10]
    root_angvel       = root_state[:, 10:13]
    
    hand_positions       = links_state[:, [10,31], 0:3]
    handles_positions    = links_state[:, [27,48], 0:3]
    
    dist_hands           =  (hand_positions - handles_positions).pow(2).sum(2).sqrt()
    
    hand_positions    = torch.flatten(hand_positions,start_dim=1)
    handles_positions = torch.flatten(handles_positions,start_dim=1)

    vel_loc, angvel_loc, roll, pitch, yaw = compute_rot_gogoro(root_orientations, inv_start_rot, root_linvel, root_angvel)

    # Get orientation information from tensors
    roll  = normalize_angle(roll).unsqueeze(-1)
    pitch = normalize_angle(pitch).unsqueeze(-1)
    yaw   = normalize_angle(yaw).unsqueeze(-1)

    # 12 index of rear wheel joint
    rear_wheel_vel = dof_vel[:, 12] / max_rear_wheel_velocity
    rear_wheel_vel = rear_wheel_vel.unsqueeze(-1)

    # 13 index of steering joint
    curr_steering_angle = dof_pos[:, 13].unsqueeze(-1)

    delta_yaw         = angvel_loc[:, 2].unsqueeze(-1)
    desired_delta_yaw = desired_delta_yaw.unsqueeze(-1)

    # Mount observation vector
    obs = torch.cat([
                        roll, pitch, yaw,
                        delta_yaw, curr_steering_angle, 
                        rear_wheel_vel, desired_delta_yaw,
                        angvel_loc[:, :2], vel_loc[:],
                        hand_positions,handles_positions,dist_hands
                        ], dim=1
                    )

    #TODO  WE SHOULD NOT apply gaussian noise like so, need a more complex model
    # # Generate gaussian noise
    # gaussian_noise = torch.randn_like(obs) * noise_scale
    
    # # Get noisy observation vector
    # noisy_obs = gaussian_noise + obs 
    # # Patch intended delta to remove noise
    # noisy_obs[:, 6] = obs[:, 6]

    return obs 




        








from isaacgym.terrain_utils import *
class Terrain:
    def __init__(self, num_robots) -> None:


        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
        self.num_per_env = 2
        self.env_length = 2.0
        self.env_width  = 2.0

        self.env_rows = 8
        self.env_cols = 8
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels  = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        #self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = 500#int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = 500#int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols))
        
        print("GENERATING PERLING TERRAIN OF SHAPE ",self.tot_rows," * ",self.tot_cols)

        noise1 = PerlinNoise(octaves=3)
        noise2 = PerlinNoise(octaves=6)
        noise3 = PerlinNoise(octaves=12)
        noise4 = PerlinNoise(octaves=24)
        print("NOISE READY")

        index = 0
        scale = 1.0
        for i in range(self.tot_rows):
            for j in range(self.tot_cols):
                noise_val = noise1([i/self.tot_rows, j/self.tot_cols])          * scale
                noise_val += 0.5 * noise2([i/self.tot_rows, j/self.tot_cols])   * scale
                noise_val += 0.25 * noise3([i/self.tot_rows, j/self.tot_cols])  * scale
                noise_val += 0.125 * noise4([i/self.tot_rows, j/self.tot_cols]) * scale
                self.height_field_raw[i,j] = noise_val
                index+=1
            print(index,"/",self.tot_rows*self.tot_cols)
        
        
        print("Perlin terrain finished")
        plt.imshow(self.height_field_raw, cmap='gray')
        plt.show()
        
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, 1.0, 1.0, 0.5)
    





@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles
