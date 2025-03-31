import numpy as np
from pathlib import Path
import torch
from perlin_noise import PerlinNoise
import random
import matplotlib.pyplot as plt
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import os
from isaacgymenvs.tasks.base.vec_task import VecTask
from gym import spaces
import xml.etree.ElementTree as ET
from collections import deque 
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# from torch.distributions import Categorical

torch.set_printoptions(precision=10)
torch.set_printoptions(sci_mode=False)

DEBUG = True

DEBUGFIXBASE = True
DEBUGUSETERRAIN  = False
DEBUG_START_SPEED = True
RANDOM_DAMPING   = False
PUSH_ROBOT       = DEBUG

CENTER_ROBOT     = DEBUG
USE_STEER_DELAY  = not DEBUG
IGNORE_ZERO      = not DEBUG



class Gogoro(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.steering_sensitivity = 0.1

        self.curent_step = 0
        #self.writer = SummaryWriter()

        self.device = rl_device
        #Sim Params
        self.n_envs             = cfg["env"]["numEnvs"]
        self.max_episode_length = torch.tensor(cfg["env"]["max_steps"])#((torch.rand(self.n_envs,device=device)+1)*cfg["env"]["max_steps"]).to(torch.long)#torch.tensor(cfg["env"]["max_steps"])



        #randomness parameters
        #self.randomization_params = cfg["task"]["randomization_params"]
        #sensors randomness
        self.imu_filter_noise       = cfg["noises"]["imu_filter_noise"]
        self.imu_noise              = cfg["noises"]["imu_noise"]
        self.speed_sensor_noise     = cfg["noises"]["speed_sensor_noise"]
        self.speed_sensor_offset    = cfg["noises"]["speed_sensor_offset"]

        #simulation randomness
        self.seating_offset  = torch.ones((self.n_envs,3))

        self.seat_offset_x_range = cfg["noises"]["seat_offset_x_range"]
        self.seat_offset_y_range = cfg["noises"]["seat_offset_y_range"]
        self.seat_offset_z_range = cfg["noises"]["seat_offset_z_range"]
        self.spawn_x_angle       = cfg["noises"]["spawn_x_angle"]
        self.imu_x_offset        = cfg["noises"]["imu_x_offset"]
        self.steering_damping_range = cfg["noises"]["steering_damping_range"]

        #control randomness
        self.steering_action_noise    = cfg["noises"]["steering_action_noise"]
        self.speed_range   = cfg["noises"]["speed_range"]
        self.speed_freq_update = cfg["noises"]["speed_freq_update"]
        self.yaw_freq_update   = cfg["noises"]["yaw_freq_update"]
        self.steering_offset = cfg["noises"]["steering_offset"]
        self.command_delay = cfg["noises"]["command_delay"]
        print("====")
        self.dof_props = None #to store the DOF props

        #curent parameters 
        self.yaw_command     = torch.zeros((self.n_envs),device=rl_device)

        #Driving limits
        self.min_speed   = 0.0
        self.max_speed   = 10.0
        self.max_steering = 0.5

        self.curent_speed         = self.get_randoms(self.n_envs,self.speed_range)
        self.last_err             = torch.zeros((self.n_envs),device=self.device)
        self.curent_command       = torch.zeros((self.n_envs),device=self.device)
        self.steer_offsets        = self.get_randoms(self.n_envs,self.steering_offset)
        self.steer_delay          = self.get_randoms(self.n_envs,self.command_delay).to(torch.long)
        self.command_history      = torch.zeros((self.n_envs,self.command_delay[1]),device=self.device)
        self.envs_indexes_        = torch.arange(0, self.n_envs)
        self.curent_damping_cfg   = self.get_randoms(self.n_envs,self.steering_damping_range)
        self.curent_speed_offset  = self.get_randoms(self.n_envs,self.speed_sensor_offset)
        self.curent_imu_x_offset  = self.get_randoms(self.n_envs,self.imu_x_offset)

        #SPEED PID PARAMS
        self.speed_no_noise = torch.zeros((self.n_envs),device=self.device)
        self.integral_error = torch.zeros((self.n_envs),device=self.device)
        self.last_err_speed = torch.zeros((self.n_envs),device=self.device)

        self.viewer = virtual_screen_capture
        self.cfg = cfg


        #Gym params
        num_obs  = 8
        num_acts = 1

        self.buff_size = 20

        self.buffer_obs       = torch.zeros((self.n_envs,self.buff_size,num_obs),device=rl_device)
        self.buffer_obs_noisy = torch.zeros((self.n_envs,self.buff_size,num_obs),device=rl_device)

        self.cfg["env"]["numObservations"]  = num_obs*self.buff_size
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

        #WRAPING THE TENSORS
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor) # PyTorch-wrapped tensor
        self.state_dof   = gymtorch.wrap_tensor(self._state_dof)


        #MAKING THE TENSORS EASIER TO USE
        self.root_positions    = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_angular_vels = self.root_tensor[:, 10:13]
        self.dof_pos = self.state_dof.view(self.n_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.state_dof.view(self.n_envs, self.num_dof, 2)[..., 1]

                
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) 

        self.root_reset_tensor  = self.root_tensor.clone().detach() # Copy root tensor
        self.root_reset_tensor[:, 7:13]  = 0 # Set all velocities to 0


        self.curent_perturbations = torch.zeros(self.n_envs,self.num_rgbd,3, device=self.device)
        self.current_steering = None

        self.reset_idx(torch.arange(0, self.n_envs).to(self.device).type(torch.long))




    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if(DEBUGUSETERRAIN):
            self._create_ground_plane()
        else:
            self._create_ground_plane_flat()
        self._create_envs()
        #self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        self.terrain = Terrain()
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices =  self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        env_spacing = 2
        envs_scale  = env_spacing * int(np.sqrt(self.n_envs))
        start_mid   = int(self.terrain.Vx_size_m/2 - envs_scale/2)

        tm_params.transform.p.x = -start_mid 
        tm_params.transform.p.y = -start_mid
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = 0.98
        tm_params.dynamic_friction = 0.98
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params) 
        self.terrain.heightsamples = self.terrain.heightsamples.to(self.device)  
    

    def _create_ground_plane_flat(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.99
        plane_params.dynamic_friction = 0.99
        #plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)




    def _create_envs(self):



        asset_root = "/home/erc/RL_NVIDIA/IsaacGymEnvs/assets"
        asset_file = "urdf/gogoro/urdf/scooter_V12.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = DEBUGFIXBASE
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = False
        asset_options.enable_gyroscopic_forces = True
        asset_options.disable_gravity = False
        asset_options.override_inertia = False # not sure what is best here, I don't think in the urdf the inertias are super good anyway
        #asset_options.linear_damping = 0.01
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)


        # Get number of DOFs and DOF names
        self.num_dof   = self.gym.get_asset_dof_count(asset)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        # Helper dictionary to map joint names to tensor ID
        self.dof_name_to_id = {k: v for k, v in zip(self.dof_names, np.arange(self.num_dof))}


        self.num_rgbd = self.gym.get_asset_rigid_body_count(asset)
        self.rgid_shape_to_id = {}
        self.rgid_body_to_id  = {}

        for i in range(self.num_rgbd):
            self.rgid_body_to_id[self.gym.get_asset_rigid_body_name(asset,i)] = i
            if(self.gym.get_asset_rigid_body_shape_indices(asset)[i].count>0):
                self.rgid_shape_to_id[self.gym.get_asset_rigid_body_name(asset,i)] = self.gym.get_asset_rigid_body_shape_indices(asset)[i].start


        self.dof_props = self.gym.get_asset_dof_properties(asset)
        #because we lock the joints to their limits we need to do first put them inside
        self.thormang_pose = torch.zeros(self.n_envs,self.num_dof, device=self.device)

        # set up the env grid
        envs_per_row = int(np.sqrt(self.n_envs))
        env_spacing  = 1.0
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        self.envs    = []
        self.handles = []
        # create and populate the environments
        for i in range(self.n_envs):


            # Everything set to pos so they all stay in-place
            for d in range(self.num_dof):
                self.dof_props['driveMode'][d] = gymapi.DOF_MODE_NONE
                self.dof_props["damping"][d]   =  0.0
                self.dof_props["stiffness"][d] =  0.0
                self.dof_props["effort"][d]    =  0.0

            #locking the thormang joints
            for j_name in self.cfg["joints_pos"]:
                id_dof = self.dof_name_to_id[j_name]
                self.dof_props['driveMode'][id_dof] = gymapi.DOF_MODE_NONE
                self.dof_props["lower"][id_dof]   =  self.cfg["joints_pos"][j_name]
                self.dof_props["upper"][id_dof]   =  self.dof_props["lower"][id_dof]+0.0001
                self.thormang_pose[:,id_dof] =  self.dof_props["lower"][id_dof]+0.0001/2


            self.dof_props['driveMode'][self.dof_name_to_id['base_x']] = gymapi.DOF_MODE_NONE
            self.dof_props["lower"][self.dof_name_to_id['base_x']]     =  0.0
            self.dof_props["upper"][self.dof_name_to_id['base_x']]     =  self.dof_props["lower"][self.dof_name_to_id['base_x']]+0.0001
            self.dof_props['driveMode'][self.dof_name_to_id['base_y']] = gymapi.DOF_MODE_NONE
            self.dof_props["lower"][self.dof_name_to_id['base_y']]     = 0.0
            self.dof_props["upper"][self.dof_name_to_id['base_y']]     = self.dof_props["lower"][self.dof_name_to_id['base_y']]+0.0001
            self.dof_props['driveMode'][self.dof_name_to_id['base_z']] = gymapi.DOF_MODE_NONE
            self.dof_props["lower"][self.dof_name_to_id['base_z']]     = 0.0
            self.dof_props["upper"][self.dof_name_to_id['base_z']]     = self.dof_props["lower"][self.dof_name_to_id['base_z']]+0.0001

            self.thormang_pose[i,self.dof_name_to_id['base_x']] = self.dof_props["lower"][self.dof_name_to_id['base_x']]+0.0001/2
            self.thormang_pose[i,self.dof_name_to_id['base_y']] = self.dof_props["lower"][self.dof_name_to_id['base_y']]+0.0001/2
            self.thormang_pose[i,self.dof_name_to_id['base_z']] = self.dof_props["lower"][self.dof_name_to_id['base_z']]+0.0001/2

            # Set velocity mode on back wheel
            self.dof_props["driveMode"][self.dof_name_to_id['rear_wheel_joint']] = gymapi.DOF_MODE_VEL
            self.dof_props["stiffness"][self.dof_name_to_id['rear_wheel_joint']] = 0.0
            self.dof_props["damping"][self.dof_name_to_id['rear_wheel_joint']] = 1000.0
            self.dof_props["effort"][self.dof_name_to_id['rear_wheel_joint']] =  170.0 #130

            self.dof_props["driveMode"][self.dof_name_to_id['steering_joint']] = gymapi.DOF_MODE_POS
            self.dof_props["stiffness"][self.dof_name_to_id['steering_joint']] =  100.0
            self.dof_props["damping"][self.dof_name_to_id['steering_joint']]   =  100.0
            self.dof_props["effort"][self.dof_name_to_id['steering_joint']]    =  10.0
            self.dof_props["velocity"][self.dof_name_to_id['steering_joint']]  =  50.0#10.0

            ref_env      = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)

            start_pose   = gymapi.Transform()
            start_pose.p = gymapi.Vec3(0.0,0.0,1.0)

            actor_handle = self.gym.create_actor(ref_env, asset, start_pose, "Gogoro", i, 1)
            actor_props  = self.gym.get_actor_rigid_shape_properties(ref_env, actor_handle)
            actor_props[self.rgid_shape_to_id["back"]].friction = 0.98
            actor_props[self.rgid_shape_to_id["back"]].rolling_friction = 0.0
            actor_props[self.rgid_shape_to_id["back"]].torsion_friction = 0.0

            actor_props[self.rgid_shape_to_id["front"]].friction         = 0.9
            actor_props[self.rgid_shape_to_id["front"]].rolling_friction = actor_props[self.rgid_shape_to_id["back"]].rolling_friction
            actor_props[self.rgid_shape_to_id["front"]].torsion_friction = actor_props[self.rgid_shape_to_id["back"]].torsion_friction


            self.gym.set_actor_rigid_shape_properties(ref_env, actor_handle, actor_props)
            self.gym.set_actor_dof_properties(ref_env, actor_handle, self.dof_props)


            #test = self.gym.get_actor_rigid_body_properties(ref_env, actor_handle)
            # for k in self.rgid_shape_to_id:
            #     i = self.rgid_shape_to_id[k]
            #     mat = test[i].inertia
            #     tensor = torch.tensor([
            #         [mat.x.x, mat.x.y, mat.x.z],
            #         [mat.y.x, mat.y.y, mat.y.z],
            #         [mat.z.x, mat.z.y, mat.z.z]
            #     ])        
                # print("RGBD ",k," = ",tensor)
                # print("==")
            #exit(0)

            self.envs.append(ref_env)
            self.handles.append(actor_handle)






    def pre_physics_step(self, actions): 

        actions = torch.clamp(actions,-1,1)
        self.curent_command = (actions[:,0]*self.max_steering)

        # TODO NOTE IMPORTANT :
        # I JUST TRIED TO USE SPEED CONTROL BUT IT IS WORKING WITHOUT
        #self.curent_command = self.curent_command+actions[:,0]*self.steering_sensitivity
        #self.curent_command = torch.clamp(self.curent_command,-1,1)*self.max_steering

        self.command_history[:,:-1]      = self.command_history[:,1:].clone()
        self.command_history[:, -1]      = self.curent_command
        if(USE_STEER_DELAY):
            applied_command = self.command_history[self.envs_indexes_,-self.steer_delay]
        else:
            applied_command = self.command_history[self.envs_indexes_,-3]

        dof_pos_command = torch.zeros(self.n_envs,self.num_dof, device=self.device)
        dof_pos_command[:,self.dof_name_to_id['steering_joint']]= applied_command#+self.steer_offsets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_pos_command))

        # target         = self.curent_speed
        # real_speed     = self.speed_no_noise
        # err_speed      = target-real_speed
        # err_deriv      = err_speed-self.last_err_speed
        # self.integral_error+=err_speed
        # self.last_err_speed = err_speed
        # self.integral_error = torch.where(self.integral_error>10.0,10.0,self.integral_error)
        # P_gain        = 1.0
        # D_gain        = 1.0
        # I_gain        = 1.0
        # PID_command    = err_speed*P_gain+ err_deriv*D_gain + self.integral_error * I_gain

        dof_speed_command = torch.zeros((self.n_envs,self.num_dof),device=self.device)
        dof_speed_command[:,self.dof_name_to_id['rear_wheel_joint']] = self.curent_speed #PD_command

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_speed_command))


                    
    def post_physics_step(self):

        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        #Compute observations
        self.compute_obs_rwd()
        curent_yaw = self.buffer_obs[:,-1,1]

        speed_command_change = (self.progress_buf==self.speed_freq_update).nonzero(as_tuple=False).flatten()
        yaw_command_change   = (self.progress_buf==self.yaw_freq_update).nonzero(as_tuple=False).flatten()
        

        if(speed_command_change.shape[0]!=0):
            speed_command_change = speed_command_change[speed_command_change]
        if(yaw_command_change.shape[0]!=0):
            yaw_command_change   = yaw_command_change[yaw_command_change]


        # print(self.curent_speed.shape)
        # print(torch.max(speed_command_change))
        # print(torch.min(speed_command_change))
        # print(speed_command_change.shape)
        self.curent_speed[speed_command_change] = self.get_randoms(len(speed_command_change),self.speed_range)

        #curent_yaw[yaw_command_change] + 0.2 * random.choice([-1,1]) +  
        self.yaw_command[yaw_command_change]    = self.get_randoms(self.n_envs,[-np.pi,np.pi])[yaw_command_change]
        self.yaw_command  = torch.where(self.yaw_command> np.pi,self.yaw_command-np.pi*2,self.yaw_command)
        self.yaw_command  = torch.where(self.yaw_command<-np.pi,self.yaw_command+np.pi*2,self.yaw_command)


        test = 0
        self.gym.clear_lines(self.viewer)
        lines  = [self.root_positions[test,0].item(), self.root_positions[test,1].item(),self.root_positions[test,2].item()+1.3,self.root_positions[test,0].item()+torch.cos(self.buffer_obs[test,-1,1]).item()*3, self.root_positions[test,1].item()+torch.sin(self.buffer_obs[test,-1,1]).item()*3,self.root_positions[test,2].item()+1.3]
        colors = [1.0, 0.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[test],1,lines,colors)

        lines  = [self.root_positions[test,0].item(), self.root_positions[test,1].item(),self.root_positions[test,2].item()+1.3,self.root_positions[test,0].item()+torch.cos(self.yaw_command[test]).item()*3, self.root_positions[test,1].item()+torch.sin(self.yaw_command[test]).item()*3,self.root_positions[test,2].item()+1.3]
        colors = [0.0, 1.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[test],1,lines,colors)

        lines  = [0, 0,1,math.cos(0)*30.0, math.sin(0)*30.0,1]
        colors = [1.0, 0.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[0],1,lines,colors)

        lines  = [0, 0,1,math.cos(np.pi)*30.0, math.sin(np.pi)*30.0,1]
        colors = [1.0, 1.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[0],1,lines,colors)

        lines  = [0, 0,1, math.cos(-np.pi)*30.0, math.sin(-np.pi)*30.0,1]
        colors = [1.0, 1.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[0],1,lines,colors)

        lines  = [0, 0,1, math.cos(-np.pi/2)*30.0, math.sin(-np.pi/2)*30.0,1]
        colors = [1.0, 1.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[0],1,lines,colors)


        if(PUSH_ROBOT):
            need_update = (self.progress_buf[:2048]+1)%10 == 0
            push_force = 30.0
            curent_yaw = curent_yaw[:2048][need_update]

            x_force = ((torch.rand(self.n_envs,device=self.device)*2-1)*push_force)[need_update]
            z_force = -((torch.rand(self.n_envs,device=self.device))*push_force)[need_update]



            self.curent_perturbations[need_update,self.rgid_body_to_id["head_p_link"],0] =  x_force*torch.cos(curent_yaw+torch.pi/2)
            self.curent_perturbations[need_update,self.rgid_body_to_id["head_p_link"],1] =  x_force*torch.sin(curent_yaw+torch.pi/2)
            self.curent_perturbations[need_update,self.rgid_body_to_id["head_p_link"],2] =  z_force


            self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(torch.flatten(self.curent_perturbations,end_dim=-2)),None)

            xpert = self.curent_perturbations[test,self.rgid_body_to_id["head_p_link"],0]
            ypert = self.curent_perturbations[test,self.rgid_body_to_id["head_p_link"],1]
            zpert = self.curent_perturbations[test,self.rgid_body_to_id["head_p_link"],2]

            nrm =torch.abs(xpert)+torch.abs(ypert)+torch.abs(zpert)

            xpert = (xpert/nrm).item()
            ypert = (ypert/nrm).item()
            zpert = (zpert/nrm).item()




            lines  = [  self.root_positions[test,0].item(),
                        self.root_positions[test,1].item(),
                        self.root_positions[test,2].item()+1.3, 
                      
                        self.root_positions[test,0].item()+xpert*3.0,
                        self.root_positions[test,1].item()+ypert*3.0,
                        self.root_positions[test,2].item()+1.3+zpert*3.0]
            
            colors = [0.0, 0.0, 1.0]

            self.gym.add_lines(self.viewer,self.envs[test],1,lines,colors)


            

            


    def compute_obs_rwd(self):
        # Refresh the tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        obs_ = compute_gogoro_observations( 
                                            self.root_tensor.clone(),
                                            self.yaw_command.clone(),
                                            self.curent_command.clone(),
                                            (self.steer_delay.clone()-self.command_delay[0])/(self.command_delay[1]-self.command_delay[0]),
                                            )

        self.buffer_obs[:,:-1]      = self.buffer_obs[:,1:].clone()
        self.buffer_obs[:,-1]       = obs_.clone()

        self.buffer_obs_noisy[:,:-1]      = self.buffer_obs_noisy[:,1:].clone()
        self.buffer_obs_noisy[:,-1]       = obs_.clone()

        self.rew_buf[:], self.reset_buf[:] = compute_gogoro_reward(
                                                                    act_buff           = self.buffer_obs[:,:,6], #buffer of actions
                                                                    curent_obs         = self.buffer_obs,
                                                                    progress_buf       = self.progress_buf,
                                                                    max_episode_length = self.max_episode_length,
                                                                    yaw_update_freq    = torch.tensor(self.yaw_freq_update),
                                                                    )
        

        self.current_steering = self.dof_pos[:,self.dof_name_to_id["steering_joint"]]


        self.curent_step+=1
        self.speed_no_noise = obs_[:,4].clone()

        #print("real speed=",self.speed_no_noise)

        self.buffer_obs_noisy[:,-1,0:2]  += self.get_randoms((self.n_envs,2),self.imu_filter_noise) #roll yaw
        self.buffer_obs_noisy[:,-1,0]    += self.curent_imu_x_offset
        self.buffer_obs_noisy[:,-1,2:4]  += self.get_randoms((self.n_envs,2),self.imu_noise)        #droll dyaw
        self.buffer_obs_noisy[:,-1,4]    += self.get_randoms(self.n_envs,self.speed_sensor_noise) #speed
        self.buffer_obs_noisy[:,-1,4]    += self.curent_speed_offset
        self.buffer_obs_noisy[:,-1,4]     = torch.where(self.buffer_obs_noisy[:,-1,4]<0,0,self.buffer_obs_noisy[:,-1,4])
        self.buffer_obs_noisy[:,-1,5]    += self.get_randoms(self.n_envs,self.imu_filter_noise) #delta yaw


        self.buffer_obs_noisy[:,-1,0:2]  /= np.pi
        self.buffer_obs_noisy[:,-1,2:4]  /= 3.0
        self.buffer_obs_noisy[:,-1,4]    /= 5.0
        self.buffer_obs_noisy[:,-1,5]    /= np.pi
        self.buffer_obs_noisy[:,-1,6]    /= self.max_steering

        #Trick to remove noise
        self.buffer_obs_noisy[:,-1,2] += self.buffer_obs[:,-2,6]-self.buffer_obs[:,-1,6]
        self.buffer_obs_noisy[:,-1,0] += (self.buffer_obs[:,-2,6]-self.buffer_obs[:,-1,6])*0.3

        self.buffer_obs_noisy[:,:,1] = 0.0
        self.obs_buf = self.buffer_obs_noisy.flatten(start_dim=1)



    def get_randoms(self,shape,bounds):
        return bounds[0] + torch.rand(shape,device=self.device) * (bounds[1] - bounds[0])

    def get_randoms_norm(self,shape,mean_cov):
        return mean_cov[0] + torch.randn(shape,device=self.device) * mean_cov[1]


        
    def randomize(self,env_ids):

        #self.apply_randomizations(self.randomization_params)
        _ids = env_ids
        n_id = _ids.shape[0]
        self.curent_speed[_ids]         = self.get_randoms(n_id,self.speed_range)
        self.steer_delay[_ids]          = self.get_randoms(n_id,self.command_delay).to(torch.long)
        self.steer_offsets[_ids]        = self.get_randoms(n_id,self.steering_offset)
        self.curent_perturbations[_ids] = torch.zeros(n_id,self.num_rgbd,3, device=self.device)
        self.curent_speed_offset[_ids]  = self.get_randoms(n_id,self.speed_sensor_offset)



    def generate_spawn_r(self,n_envs):
        initial_target  = torch.zeros(n_envs,device=self.device)#(torch.rand(n_envs,device=self.device)*2-1)*np.pi #torch.zeros(n_envs,device=self.device) #(torch.rand(n_envs,device=self.device)*2-1)*np.pi
        initial_rot     = torch.zeros(n_envs,device=self.device) #initial_target +  self.get_randoms(n_envs,[-np.pi,np.pi]) #(torch.rand(n_envs,device=self.device)*2-1)*np.pi #initial_target
        rot             = self.euler_to_quaternion( torch.zeros(n_envs,device=self.device),#(torch.rand(n_envs,device=self.device)*2-1)*0.1,
                                                    torch.zeros(n_envs,device=self.device),
                                                    initial_rot)
        

        
        # initial_target  = torch.zeros(n_envs,device=self.device)
        # initial_rot     = torch.zeros(n_envs,device=self.device)
        # rot             = self.euler_to_quaternion( torch.zeros(n_envs,device=self.device),
        #                                             torch.zeros(n_envs,device=self.device),
        #                                             initial_rot)


        return rot, initial_target, initial_rot



    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
        qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
        qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
        qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
        q = torch.stack((qx, qy, qz, qw), dim=-1)
        return q


    def reset_idx(self, env_ids):




        nbresets = len(env_ids)
        if(nbresets==0):
            return

        env_ids_int32  = env_ids.to(dtype=torch.int32)
        _env_ids_int32 = gymtorch.unwrap_tensor(env_ids_int32) 


        self.randomize(env_ids_int32)
        r, spawn_yaw_tgt, initial_rot = self.generate_spawn_r(nbresets)
        



        test = self.root_reset_tensor.clone()
        test[env_ids_int32,2] = 0.03
        if(DEBUGUSETERRAIN):
            #            test[env_ids_int32,2] += self.terrain.heightsamples[int(1024/2)-32+env_ids%64,int(1024/2)-32+env_ids//64] * self.terrain.H_scale
            
            #env_spacing = 1 meter

            sqrt_root   = int(np.sqrt(self.n_envs))
            env_spacing = 2
            envs_scale  = env_spacing * sqrt_root
            start_mid   = int(self.terrain.Vx_size_m/2 - envs_scale/2)
            start_mid_index = int(start_mid * (1/self.terrain.V_scale))

            test[env_ids_int32,2] += self.terrain.heightsamples[start_mid_index+((env_ids_int32 %sqrt_root) * (env_spacing/self.terrain.V_scale)).to(torch.int) ,
                                                                start_mid_index+((env_ids_int32//sqrt_root) * (env_spacing/self.terrain.V_scale)).to(torch.int)] * self.terrain.H_scale
            

        test[env_ids_int32,3:7]  = r
        test[env_ids_int32,7:13] = 0.0

        if(DEBUG_START_SPEED):
            start_speed = 1.3
            test[env_ids_int32,7] = start_speed*torch.cos(initial_rot)
            test[env_ids_int32,8] = start_speed*torch.sin(initial_rot)



        assert self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(test),_env_ids_int32, len(env_ids_int32))

        self.dof_pos[env_ids_int32] = self.thormang_pose[env_ids_int32]
        self.dof_vel[env_ids_int32] = 0.0

        assert self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.state_dof),_env_ids_int32, len(env_ids_int32))


        for id in env_ids:
            
            self.curent_imu_x_offset[id]  = self.get_randoms(1,self.imu_x_offset)

            if(RANDOM_DAMPING):
                self.curent_damping_cfg[id] = self.get_randoms(1,self.steering_damping_range)
                self.set_env_dof_prop(id,self.curent_damping_cfg[id],13700.0, "steering_joint")
                #self.set_env_dof_prop(id,5000.0,50000.0, "steering_joint")


            if(not CENTER_ROBOT):
                self.dof_props['driveMode'][self.dof_name_to_id['base_x']] = gymapi.DOF_MODE_NONE
                self.dof_props["lower"][self.dof_name_to_id['base_x']]     =  self.get_randoms(1,self.seat_offset_x_range).item()
                self.dof_props["upper"][self.dof_name_to_id['base_x']]     =  self.dof_props["lower"][self.dof_name_to_id['base_x']]+0.0001
                self.dof_props['driveMode'][self.dof_name_to_id['base_y']] = gymapi.DOF_MODE_NONE
                self.dof_props["lower"][self.dof_name_to_id['base_y']]     = self.get_randoms(1,self.seat_offset_y_range).item()
                self.dof_props["upper"][self.dof_name_to_id['base_y']]     = self.dof_props["lower"][self.dof_name_to_id['base_y']]+0.0001
                self.dof_props['driveMode'][self.dof_name_to_id['base_z']] = gymapi.DOF_MODE_NONE
                self.dof_props["lower"][self.dof_name_to_id['base_z']]     = self.get_randoms(1,self.seat_offset_z_range).item()
                self.dof_props["upper"][self.dof_name_to_id['base_z']]     = self.dof_props["lower"][self.dof_name_to_id['base_z']]+0.0001

            if(RANDOM_DAMPING or not CENTER_ROBOT):
                self.gym.set_actor_dof_properties(self.envs[id], self.handles[id], self.dof_props)

        # # Reset buffers
        self.progress_buf[env_ids_int32]        = 0
        self.reset_buf[env_ids_int32]           = 0
        self.obs_buf[env_ids_int32]             = 0
        self.buffer_obs[env_ids_int32]          = 0.0
        self.buffer_obs_noisy[env_ids_int32]    = 0.0
        self.curent_command[env_ids_int32]      = 0.0

        self.yaw_command[env_ids_int32]         = spawn_yaw_tgt.clone()
        self.command_history[env_ids_int32]     = 0.0
        self.last_err[env_ids_int32]            = 0.0
        self.speed_no_noise[env_ids_int32]      = 0.0
        self.integral_error[env_ids_int32]      = 0.0
        

    def set_env_dof_prop(self,env_id,damping,stiffness, joint_name):
        self.dof_props["driveMode"][self.dof_name_to_id[joint_name]] = gymapi.DOF_MODE_POS
        self.dof_props["stiffness"][self.dof_name_to_id[joint_name]] =  stiffness
        self.dof_props["damping"][self.dof_name_to_id[joint_name]]   =  damping
        self.dof_props["effort"][self.dof_name_to_id[joint_name]]    =  200.0
        self.dof_props["velocity"][self.dof_name_to_id[joint_name]]  =  1.00#0.5 
        self.gym.set_actor_dof_properties(self.envs[env_id], self.handles[env_id], self.dof_props)

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def KL_uniform(x):

    prob_actions         = F.softmax(x, dim=1)
    uniform_distribution = torch.ones_like(prob_actions)/prob_actions.shape[1]
    kl_divergence        = torch.sum(prob_actions * torch.log(prob_actions/uniform_distribution), dim=1)

    return kl_divergence

@torch.jit.script
def compute_gogoro_reward(act_buff,curent_obs, progress_buf, max_episode_length, yaw_update_freq):
    tilt      = curent_obs[:,-1, 0]
    dtilt     = curent_obs[:,-1, 2]
    dyaw      = curent_obs[:,: , 3]
    speed     = curent_obs[:,-1, 4]
    yaw_err   = curent_obs[:,-1, 5]


    act_buff_ = act_buff/0.5

    max_tilt    = 0.38#1.0
    tilt_err = tilt/max_tilt
    tilt_err = torch.clamp(tilt_err,-1.0,1.0)

    yaw_err = yaw_err/np.pi
    yaw_err = torch.clamp(yaw_err,-1.0,1.0)

    dtilt_err = dtilt/0.3
    dtilt_err = torch.clamp(dtilt_err,-1.0,1.0)

    dyaw_err = dyaw
    dyaw_err = torch.clamp(dyaw_err,-1.0,1.0)

    reward1 = 1-yaw_err**2
    reward2 = 1-tilt_err**2
    reward4 = 1-dtilt_err**2


    tilt_w      = 1-torch.tanh(50*tilt_err**2)
    dtilt_w     = 1-torch.tanh(50*dtilt_err**2)

    reward5    = 1-((act_buff_[:,-1]**2) * (tilt_w * dtilt_w) )

    reward7    = 1-torch.mean(torch.diff(act_buff_,dim=1)**2)

 
    #this make a stable policy but it doesnt want to turn
    #reward     = reward1*0.3 + reward2*0.1 + reward4 * 0.35 + reward5 * 2.0 + reward6 * 3.0 + reward7 * 0.2

    reward     = reward1*0.45 + reward2*0.1 + reward4 * 0.35 + reward5 * 2.0 + reward7 * 0.2

    finished   = progress_buf >= max_episode_length - 1

    felt       = torch.abs(tilt)>= max_tilt
    reset      = torch.where(finished, 1, 0)
    reset      = torch.where(felt    ,1, reset) 
    reward = torch.clip(reward, 0., None)
    reward = torch.where(felt  ,-1, reward) 
    return reward.detach(), reset


@torch.jit.script
def shortest_angle_distance(yaw1, yaw2):
    return (yaw2 - yaw1 + torch.pi)%(2*torch.pi) - torch.pi


@torch.jit.script
def compute_gogoro_observations(root_state,desired_yaw,curent_command, steer_delay):
    
    root_orientations = root_state[:, 3:7]

    roll, _, yaw = get_euler_xyz(root_orientations)

    base_lin_vel = quat_rotate_inverse(root_orientations, root_state[:, 7:10])
    base_ang_vel = quat_rotate_inverse(root_orientations, root_state[:, 10:13])

    d_roll  = base_ang_vel[:,0]
    d_yaw   = base_ang_vel[:,2]

    roll = torch.where(roll>np.pi, roll-np.pi*2,roll)
    roll = torch.where(roll<-np.pi,roll+np.pi*2,roll)

    yaw  = torch.where(yaw>np.pi, yaw-np.pi*2,yaw)
    yaw  = torch.where(yaw<-np.pi,yaw+np.pi*2,yaw)

    delta_yaw = shortest_angle_distance(yaw, desired_yaw)  

    speed = base_lin_vel[:,0]

    obs = torch.cat([
                        roll.unsqueeze(-1), #0
                        yaw.unsqueeze(-1),  #1
                        d_roll.unsqueeze(-1), #2
                        d_yaw.unsqueeze(-1), #3
                        speed.unsqueeze(-1) , #4
                        delta_yaw.unsqueeze(-1), #5
                        curent_command.unsqueeze(-1),#6
                        steer_delay.unsqueeze(-1), #7
                        ], dim=1
                    )

    return obs 




        








from isaacgym.terrain_utils import *
import matplotlib.pyplot as plt
import math
class Terrain:
    def __init__(self) -> None:
        
        self.Vx_shape = 512 
        self.Vy_shape = 512 

        self.V_scale = 0.5 #2.0
        self.H_scale = 1.0#1.0/4 #2.0

        self.Vx_size_m = self.V_scale*self.Vx_shape
        self.Vy_size_m = self.V_scale*self.Vy_shape


        self.heightsamples = self.rand_perlin_2d_octaves((self.Vx_shape, self.Vy_shape), (1, 4), 2)

        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.heightsamples.numpy(), self.V_scale, self.H_scale, None)


    def rand_perlin_2d(self,shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    def rand_perlin_2d_octaves(self,shape, res, octaves=1, persistence=0.5):
        noise = torch.zeros(shape)
        frequency = 2
        amplitude = 1 #1
        for _ in range(octaves):
            noise += amplitude * self.rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
            frequency *= 2
            amplitude *= persistence
        return noise







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
