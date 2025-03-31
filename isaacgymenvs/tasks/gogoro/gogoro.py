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
from pathlib import Path
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


USEPROCEDURALPLANE = False
DEBUG = False
DEBUGFIXBASE = False
DEBUGUSEPRISMATIC = True
DEBUGNOSPEED = False
DEBUGNOSTEER = False
DEBUGNOYAW   = True
DEBUG_USE_IK = False


class Gogoro(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        

        self.viewer = virtual_screen_capture
        self.cfg = cfg
        
        #Driving parameters
        self.tilt_limit  = torch.tensor(0.02)
        self.min_speed   = torch.tensor(self.cfg["env"]["speedRange"][0])
        self.max_speed   = torch.tensor(self.cfg["env"]["speedRange"][1])
        self.min_yaw = torch.tensor(np.radians(self.cfg["env"]["yawRange"][0]))
        self.max_yaw = torch.tensor(np.radians(self.cfg["env"]["yawRange"][1]))

        #Sim Params
        self.n_envs = self.cfg["env"]["numEnvs"]
        self.max_episode_length = torch.tensor(self.cfg["env"]["maxEpisodeLength"])

        #Gym params
        self.obs_memory = 1
        self.obs_shape = 8
        self.curent_obs = torch.zeros((self.n_envs,self.obs_shape),device=rl_device)
        num_obs = self.obs_shape * self.obs_memory
        num_acts = 1
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

        #WRAPING THE TENSORS
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor) # PyTorch-wrapped tensor
        self.state_dof   = gymtorch.wrap_tensor(self._state_dof)
        self.rb_states   = gymtorch.wrap_tensor(self._rb_states)
        self.jacobian    = gymtorch.wrap_tensor(_jacobian)           # jacobian shape : num_envs, num_links, 6, num_dofs + 6


        #MAKING THE TENSORS EASIER TO USE
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

        self.leftDOFIndex =np.array([18,19,20,21,22,23,24])
        self.rightDOFIndex=np.array([27,28,29,30,31,32,33])
        self.rgdb_to_index      = self.gym.get_actor_rigid_body_dict(self.envs[-1],self.handles[-1]) #its the same for all the envs anyway
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step

        self.root_reset_tensor  = self.root_tensor.clone() # Copy root tensor
        self.reset_state_dof   = self.state_dof.clone()
        self.default_dof_pos   = self.dof_pos.clone()
        self.root_reset_tensor[:, 7:13]  = 0 # Set all velocities to 0
        self.rigid_body_reset = self.rb_states.clone()
        
        self.reset_state_dof = self.reset_state_dof.view(self.num_envs, self.num_dof, 2)
        # Set DOF initial position according to CFG
        for j_name in self.cfg["joints_pos"]:
            print(j_name)
            print(self.reset_state_dof.shape)
            self.reset_state_dof[:,self.dof_name_to_id[j_name],0]  = self.cfg["joints_pos"][j_name]
            self.reset_state_dof[:,self.dof_name_to_id[j_name],1]  = 0 # all vel to 0
        self.reset_state_dof.view(self.state_dof.shape)

        

        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.target_yaw     = torch.zeros((self.n_envs),device=self.device)
        self.curent_speed   = torch.rand((self.n_envs),device=self.device)
        self.last_steering_action = torch.zeros((self.n_envs),device=self.device)
        
        self.prismaticvals = (-0.06-0.06)*torch.rand(self.n_envs,5).to(self.device) + 0.06
        self.reset_idx(torch.arange(0, self.n_envs).to(self.device).type(torch.long))

    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if(not USEPROCEDURALPLANE):
            self._create_ground_plane()
        else:
            self._create_ground_realistic()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.99
        plane_params.dynamic_friction = 0.99
        #plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_ground_realistic(self):
        
        
        self.terrain = Terrain(num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices =  self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -20
        tm_params.transform.p.y = -250
        tm_params.transform.p.z = -0.15
        tm_params.static_friction = 0.9
        tm_params.dynamic_friction = 0.9
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "/home/erc/RL_NVIDIA/IsaacGymEnvs/assets"
        asset_file = "urdf/gogoro/urdf/gogoro_and_thormang3_Light_freewheels.urdf"#

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = DEBUGFIXBASE
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = False
        asset_options.enable_gyroscopic_forces = True
        asset_options.slices_per_cylinder = 1000
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
            dof_props['damping'][i]   = 300.0


        # Set velocity mode on back wheel
        dof_props["driveMode"][self.dof_name_to_id['rear_wheel_joint']] = gymapi.DOF_MODE_VEL
        dof_props["stiffness"][self.dof_name_to_id['rear_wheel_joint']] = 0.0
        dof_props["damping"][self.dof_name_to_id['rear_wheel_joint']] = 3.0

        # Front wheel mode set to None so it runs freely
        dof_props["driveMode"][self.dof_name_to_id['front_wheel_joint']] = gymapi.DOF_MODE_NONE
        dof_props["stiffness"][self.dof_name_to_id['front_wheel_joint']] = 0
        dof_props["damping"][self.dof_name_to_id['front_wheel_joint']] = 0

        #joints of the freewheels
        dof_props["driveMode"][self.dof_name_to_id['l_metal_freewheel_holder_TO_l_dummy']] = gymapi.DOF_MODE_NONE
        dof_props["stiffness"][self.dof_name_to_id['l_metal_freewheel_holder_TO_l_dummy']] = 0
        dof_props["damping"][self.dof_name_to_id['l_metal_freewheel_holder_TO_l_dummy']] = 0
        dof_props["effort"][self.dof_name_to_id['l_metal_freewheel_holder_TO_l_dummy']] = 0
        dof_props["friction"][self.dof_name_to_id['l_metal_freewheel_holder_TO_l_dummy']] = 0.0001

        dof_props["driveMode"][self.dof_name_to_id['dummy_TO_l_free_wheel']] = gymapi.DOF_MODE_NONE
        dof_props["stiffness"][self.dof_name_to_id['dummy_TO_l_free_wheel']] = 0
        dof_props["damping"][self.dof_name_to_id['dummy_TO_l_free_wheel']] = 0
        dof_props["effort"][self.dof_name_to_id['dummy_TO_l_free_wheel']] = 0
        dof_props["friction"][self.dof_name_to_id['dummy_TO_l_free_wheel']] = 0.0001

        dof_props["driveMode"][self.dof_name_to_id['r_metal_freewheel_holder_TO_r_dummy']] = gymapi.DOF_MODE_NONE
        dof_props["stiffness"][self.dof_name_to_id['r_metal_freewheel_holder_TO_r_dummy']] = 0
        dof_props["damping"][self.dof_name_to_id['r_metal_freewheel_holder_TO_r_dummy']] = 0
        dof_props["effort"][self.dof_name_to_id['r_metal_freewheel_holder_TO_r_dummy']] = 0
        dof_props["friction"][self.dof_name_to_id['r_metal_freewheel_holder_TO_r_dummy']] = 0.0001

        dof_props["driveMode"][self.dof_name_to_id['dummy_TO_r_free_wheel']] = gymapi.DOF_MODE_NONE
        dof_props["stiffness"][self.dof_name_to_id['dummy_TO_r_free_wheel']] = 0
        dof_props["damping"][self.dof_name_to_id['dummy_TO_r_free_wheel']] = 0
        dof_props["effort"][self.dof_name_to_id['dummy_TO_r_free_wheel']] = 0
        dof_props["friction"][self.dof_name_to_id['dummy_TO_r_free_wheel']] = 0.0001



        # Set position mode for steering joint
        dof_props["driveMode"][self.dof_name_to_id['steering_joint']] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][self.dof_name_to_id['steering_joint']] =  1000.0
        dof_props["damping"][self.dof_name_to_id['steering_joint']]   =  100.0
        
        
        #set position of the hands
        if(DEBUGUSEPRISMATIC):
            # dof_props["driveMode"][self.dof_name_to_id['l_handle_prismatic_joint']] = gymapi.DOF_MODE_POS
            # dof_props["stiffness"][self.dof_name_to_id['l_handle_prismatic_joint']] =  10000.0
            # dof_props["damping"][self.dof_name_to_id[  'l_handle_prismatic_joint']]   =  10.0
            # dof_props["driveMode"][self.dof_name_to_id['r_handle_prismatic_joint']] = gymapi.DOF_MODE_POS
            # dof_props["stiffness"][self.dof_name_to_id['r_handle_prismatic_joint']] =  10000.0
            # dof_props["damping"][self.dof_name_to_id[  'r_handle_prismatic_joint']]   =  10.0
            dof_props["driveMode"][self.dof_name_to_id['base_x']] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][self.dof_name_to_id['base_x']] =  10000000.0
            dof_props["damping"][self.dof_name_to_id[  'base_x']]   =  10.0
            dof_props["driveMode"][self.dof_name_to_id['base_y']] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][self.dof_name_to_id['base_y']] =  10000000.0
            dof_props["damping"][self.dof_name_to_id[  'base_y']]   =  10.0
            dof_props["driveMode"][self.dof_name_to_id['base_z']] = gymapi.DOF_MODE_POS
            dof_props["stiffness"][self.dof_name_to_id['base_z']] =  10000000.0
            dof_props["damping"][self.dof_name_to_id[  'base_z']]   =  10.0


        # set up the env grid
        envs_per_row = num_per_row
        env_spacing  = spacing
        env_lower    = gymapi.Vec3(-env_spacing, 0.0,        -env_spacing)
        env_upper    = gymapi.Vec3( env_spacing, env_spacing, env_spacing)

        # Set the target position of the DOFs according to vthe cfg
        self.pos_targets = np.zeros(self.num_dof).astype('f')
        # Set DOF initial position according to CFG
        for j_name in self.cfg["joints_pos"]:
            self.pos_targets[self.dof_name_to_id[j_name]] = self.cfg["joints_pos"][j_name]


        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.1)
        start_pose.r = gymapi.Quat.from_euler_zyx(-0.3, 0, 0)#gymapi.Quat(0.0, 0.0, 0.0, 1.0)#gymapi.Quat.from_euler_zyx(0, 0, 1.57)#
        _0_rot        = gymapi.Quat.from_euler_zyx(0, 0, 0)
        self.start_rotation = torch.tensor([_0_rot.x, _0_rot.y, _0_rot.z, _0_rot.w], device=self.device)

        self.envs = []
        self.handles = []
        # create and populate the environments
        for i in range(num_envs):
            ref_env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            actor_handle = self.gym.create_actor(ref_env, asset, start_pose, "Gogoro", i, 1)
            actor_props = self.gym.get_actor_rigid_shape_properties(ref_env, actor_handle)
            
            # actor_props[0].friction = 0.99
            # actor_props[0].rolling_friction = 0.0
            # actor_props[0].torsion_friction = 0.0

            # actor_props[1].friction = 0.0
            # actor_props[1].rolling_friction = 0.0
            # actor_props[1].torsion_friction = 0.0


            # actor_props[2].friction = 0.99
            # actor_props[2].rolling_friction = 0.0
            # actor_props[2].torsion_friction = 0.0


            # actor_props[3].friction = 0.99
            # actor_props[3].rolling_friction = 0.0
            # actor_props[3].torsion_friction = 0.0


            # rgdb_to_index = self.gym.get_actor_rigid_body_dict(ref_env,actor_handle)

            # actor_props[rgdb_to_index["r_free_wheel"]].friction = .01
            # actor_props[rgdb_to_index["r_free_wheel"]].rolling_friction = .01
            # actor_props[rgdb_to_index["r_free_wheel"]].torsion_friction = .01

            # actor_props[rgdb_to_index["l_free_wheel"]].friction = .01
            # actor_props[rgdb_to_index["l_free_wheel"]].rolling_friction = .01
            # actor_props[rgdb_to_index["l_free_wheel"]].torsion_friction = .01

            self.gym.set_actor_rigid_shape_properties(ref_env, actor_handle, actor_props)
            
            self.envs.append(ref_env)
            self.handles.append(actor_handle)
            self.gym.set_actor_dof_properties(ref_env, actor_handle, dof_props)
            self.gym.set_actor_dof_position_targets(ref_env, actor_handle, self.pos_targets)

    def pre_physics_step(self, actions):    
        #create array conatining the velocities
        torch_speeds = torch.zeros((self.num_envs,self.num_dof),device=self.device)
        #setting the rear wheel vel to 1000
    
        wheel_spin_delay = 20
        time_to_idl      = 50

        speed = torch.where(
                                self.progress_buf<  (wheel_spin_delay+time_to_idl),
                                self.curent_speed * ((self.progress_buf-wheel_spin_delay)/(wheel_spin_delay+time_to_idl)),
                                self.curent_speed
                            )
        
        # speed = torch.where(
        #                         self.progress_buf<wheel_spin_delay,
        #                         speed*0.0,
        #                         speed
        #                     )
                
        speed_scaled = ((speed)*30)+20
        if(DEBUGNOSPEED):
            speed_scaled*=0

        torch_speeds[:,self.dof_name_to_id['rear_wheel_joint']] = speed_scaled

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
        
        offset=6
        if(DEBUGFIXBASE):
            offset=0
        j_eefR = self.jacobian[:, right_hand_index, :, self.rightDOFIndex+offset ]
        j_eefL = self.jacobian[:, left_hand_index, :, self.leftDOFIndex+offset   ]
        
        test = self.dof_pos.clone()
        if(DEBUG_USE_IK):
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

            lmbdaR = torch.eye(6, device=self.device) * (0.3 ** 2)
            lmbdaL = torch.eye(6, device=self.device) * (0.3 ** 2)
            
            uR = (j_eefR_T @ torch.inverse(j_eefR @ j_eefR_T + lmbdaR) @ Rdpose).view(self.n_envs, 7)
            uL = (j_eefL_T @ torch.inverse(j_eefL @ j_eefL_T + lmbdaL) @ Ldpose).view(self.n_envs, 7)

            test[:,self.rightDOFIndex]+=uR
            test[:,self.leftDOFIndex] +=uL

        self.last_steering_action = actions[:,0]
        steer = actions[:,0]*self.action0_scale
        if(DEBUGNOSTEER):
            steer*=0
        
        test[:,self.dof_name_to_id['steering_joint']]= steer
        if(DEBUGUSEPRISMATIC):
            test[:,self.dof_name_to_id['r_handle_prismatic_joint']] = self.prismaticvals[:,0]
            test[:,self.dof_name_to_id['l_handle_prismatic_joint']] = self.prismaticvals[:,1]
            test[:,self.dof_name_to_id['base_x']] = self.prismaticvals[:,2]
            test[:,self.dof_name_to_id['base_y']] = self.prismaticvals[:,3]
            test[:,self.dof_name_to_id['base_z']] = self.prismaticvals[:,4]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(test))
        

        if(DEBUG):
            pass
            # torso_quat       = quat_mul(self.root_orientations,self.inv_start_rot)
            # _, _, yaw = get_euler_xyz(torso_quat)
            # self.gym.clear_lines(self.viewer)
            # env_debug = 0
            # l=2
            # point1 = self.root_positions[env_debug].clone().cpu().numpy()
            # point1[2] += 1
            # point2 = np.array(
            #                     [
            #                         point1[0]+l*math.cos(yaw[env_debug]),
            #                         point1[1]+l*math.sin(yaw[env_debug]),
            #                         point1[2]
            #                     ]
            #                 )
            
            # point3 = np.array(
            #                 [
            #                         point1[0]+l*math.cos(self.target_yaw[env_debug]),
            #                         point1[1]+l*math.sin(self.target_yaw[env_debug]),
            #                         point1[2]
            #                 ]
            #                 )
            # #print("TEST = ",self.root_orientations[env_debug,2])
            # line_1 = np.concatenate((point1,  point2),  axis=0).astype(np.float32)
            # line_2 = np.concatenate((point1,  point3),  axis=0).astype(np.float32)
            # self.gym.add_lines(self.viewer,self.envs[env_debug],1,line_1,np.array((1.0,0.0,0.0),dtype=np.float32))
            # self.gym.add_lines(self.viewer,self.envs[env_debug],1,line_2,np.array((0.0,1.0,0.0),dtype=np.float32))
    
    def post_physics_step(self):

        self.progress_buf += 1
        # Check which environments should be reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # # Compute observations
        self.compute_observations()
        
        # # Compute rewards
        self.compute_reward()

            

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_gogoro_reward(
                                                                    self.curent_obs,
                                                                    self.progress_buf,
                                                                    self.reset_buf,
                                                                    self.max_episode_length,
                                                                    )




    def compute_observations(self):
        # Refresh the tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if(DEBUG_USE_IK):
            self.gym.refresh_jacobian_tensors(self.sim)


        # Randomly generate offsets for the target delta yaw from a gaussian
        # distribution
        random_offsets = torch.rand((self.n_envs),device=self.device)  * (self.max_yaw - self.min_yaw) + self.min_yaw
        
        curent_yaw     = self.curent_obs[:,2]
        
        self.can_update_goal = self.progress_buf % 150 == 0
        self.target_reached  = torch.abs(self.target_yaw-curent_yaw) < 0.1
        self.can_update_goal = self.can_update_goal * self.target_reached
        
        
        if(DEBUGNOYAW):
            self.target_yaw = torch.zeros((self.n_envs),device=self.device)
        else:
            self.target_yaw = torch.where(
                                            self.can_update_goal, 
                                            curent_yaw + random_offsets * random.choice([1,-1]),
                                            self.target_yaw
                                            )

            self.target_yaw = torch.where(
                                            self.progress_buf == 0, 
                                            random_offsets * random.choice([1,-1]),
                                            self.target_yaw
                                            )
            

        # self.curent_speed   = torch.where(
        #                                     self.progress_buf % 150 == 0, 
        #                                     torch.rand((self.n_envs),device=self.device)*(1-0.6)+0.6,
        #                                     self.curent_speed
        #                                 )
        
        self.curent_speed = torch.where(
                                        self.progress_buf == 0, 
                                        torch.rand((self.n_envs),device=self.device)*(1-0.6)+0.6,
                                        self.curent_speed
                                        )

        self.curent_obs = compute_gogoro_observations( 
                                                    self.root_tensor,
                                                    self.target_yaw,
                                                    self.last_steering_action,
                                                    self.inv_start_rot
                                                    )
        
        if(self.obs_memory>1):
            self.obs_buf[:,:-self.obs_shape] = self.obs_buf[:,self.obs_shape:]
            self.obs_buf[:,-self.obs_shape:] = self.curent_obs
        else:
            self.obs_buf  = self.curent_obs
        
        
    def reset_idx(self, env_ids):
                
        env_ids_int32  = env_ids.to(dtype=torch.int32)
        _env_ids_int32 = gymtorch.unwrap_tensor(env_ids_int32) 

        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                    gymtorch.unwrap_tensor(self.root_reset_tensor),
                                                    _env_ids_int32, len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.reset_state_dof),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



        if(DEBUGUSEPRISMATIC):
            self.prismaticvals[env_ids] = (-0.06-0.06)*torch.rand(len(env_ids_int32),5).to(self.device) + 0.06

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.reset_state_dof),
                                                    _env_ids_int32, len(env_ids_int32))
        
        
        # # Reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.obs_buf[env_ids,:] = 0
        self.curent_obs[env_ids,:] = 0
        
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
def compute_gogoro_reward(curent_obs, progress_buf, reset_buf, max_episode_length):#, delta_scale):
    # # Reset if over iteration limit
    reset = torch.where(progress_buf > max_episode_length, 1, reset_buf)

    tilt = curent_obs[:, 0]

    pitch = curent_obs[:, 1]


    reset = torch.where(progress_buf > 500, 1, 0)
    reset = torch.where(torch.abs(tilt) > 1, 1, reset)
    reset = torch.where(torch.abs(pitch) > 0.1, 1, reset)

    reward = -(tilt**2) * 100

    return reward, reset


@torch.jit.script
def compute_reward(curent_obs):
    tilt = curent_obs[:,:, 0]
    reward = -(tilt**2) * 100
    return reward






@torch.jit.script
def compute_gogoro_observations(root_state, desired_yaw,last_steering_action,inv_start_rot):
    
    root_orientations = root_state[:, 3:7]
    root_linvel       = root_state[:, 7:10]
    root_angvel       = root_state[:, 10:13]

    torso_quat       = quat_mul(root_orientations,inv_start_rot)
    roll, pitch, yaw = get_euler_xyz(torso_quat)
    base_ang_vel = quat_rotate_inverse(torso_quat, root_angvel)


    roll = torch.where(roll>np.pi,roll-np.pi*2,roll)
    pitch = torch.where(pitch>np.pi,pitch-np.pi*2,pitch)
    yaw = torch.where(yaw>np.pi,yaw-np.pi*2,yaw)

    # Get orientation information from tensors
    delta_yaw = yaw-desired_yaw    
    
    speed = torch.sum(torch.abs(root_linvel[:,:2]),dim=1)*3.6 #ms to kmH
    obs = torch.cat([
                        roll.unsqueeze(-1),
                        pitch.unsqueeze(-1),
                        yaw.unsqueeze(-1),
                        delta_yaw.unsqueeze(-1),
                        speed.unsqueeze(-1)/100,
                        base_ang_vel
                        ], dim=1
                    )

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

        self.border   = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols))
        
        SAVE_CACHE = False
        cache = Path("./cache/h.npy")
        if cache.exists():
            print("[using cache] ./cache/h.npy") 
            self.height_field_raw = np.load("./cache/h.npy")
        else:
            print("[no cache] ./cache doesnt exist") 
            SAVE_CACHE = True
            
            print("GENERATING PERLING TERRAIN OF SHAPE ",self.tot_rows," * ",self.tot_cols)

            noise1 = PerlinNoise(octaves=3)
            noise2 = PerlinNoise(octaves=6)
            noise3 = PerlinNoise(octaves=12)
            noise4 = PerlinNoise(octaves=24)
            print("NOISE READY")

            index = 0
            scale = 10.0
            for i in range(self.tot_rows):
                for j in range(self.tot_cols):
                    noise_val = noise1([i/self.tot_rows, j/self.tot_cols])          * scale
                    noise_val += 0.5 * noise2([i/self.tot_rows, j/self.tot_cols])   * scale
                    noise_val += 0.25 * noise3([i/self.tot_rows, j/self.tot_cols])  * scale
                    noise_val += 0.125 * noise4([i/self.tot_rows, j/self.tot_cols]) * scale
                    self.height_field_raw[i,j] = noise_val
                    index+=1
                print(index,"/",self.tot_rows*self.tot_cols)
            
            if(SAVE_CACHE):
                np.save("./cache/h.npy",self.height_field_raw)
            
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
