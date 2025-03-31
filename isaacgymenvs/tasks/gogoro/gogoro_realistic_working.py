import numpy as np
from pathlib import Path
import torch
from perlin_noise import PerlinNoise
import random
import matplotlib.pyplot as plt
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
from gym import spaces
import xml.etree.ElementTree as ET
from collections import deque 

torch.set_printoptions(precision=2)
torch.set_printoptions(sci_mode=False)

DEBUG = False

DEBUGFIXBASE = False
DEBUGUSETERRAIN  = False
DEBUG_YAW_RANDOM = False

PUSH_ROBOT       = not DEBUG
CENTER_ROBOT     = DEBUG
RANDOM_DAMPING   = not DEBUG


class Gogoro(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):


        self.device = rl_device
        #Sim Params
        self.n_envs             = cfg["env"]["numEnvs"]
        self.max_episode_length = torch.tensor(cfg["env"]["max_steps"])



        #randomness parameters
        self.randomization_params = cfg["task"]["randomization_params"]
        #sensors randomness
        self.imu_filter_noise       = cfg["noises"]["imu_filter_noise"]
        self.imu_noise              = cfg["noises"]["imu_noise"]
        self.speed_sensor_noise     = cfg["noises"]["speed_sensor_noise"]

        #simulation randomness
        self.seating_offset  = torch.ones((self.n_envs,3))

        self.seat_offset_x_range = cfg["noises"]["seat_offset_x_range"]
        self.seat_offset_y_range = cfg["noises"]["seat_offset_y_range"]
        self.seat_offset_z_range = cfg["noises"]["seat_offset_z_range"]
        self.spawn_x_angle = cfg["noises"]["spawn_x_angle"]
        self.steering_damping_range = cfg["noises"]["steering_damping_range"]

        #control randomness
        self.steering_action_noise    = cfg["noises"]["steering_action_noise"]
        self.speed_range   = cfg["noises"]["speed_range"]
        self.speed_freq_update = cfg["noises"]["speed_freq_update"]
        self.yaw_freq_update   = cfg["noises"]["yaw_freq_update"]
        self.steering_offset = cfg["noises"]["steering_offset"]

        self.command_delay = cfg["noises"]["command_delay"]


        #curent parameters 
        self.yaw_command     = torch.zeros((self.n_envs),device=rl_device)


        #Driving limits
        self.tilt_limit  = 0.02
        self.min_speed   = 0.0
        self.max_speed   = 10.0
        self.max_steering = 0.3


        self.curent_speed         = self.get_randoms(self.n_envs,self.speed_range)
        self.curent_command       = torch.zeros((self.n_envs,1),device=self.device)
        self.steering_speed       = torch.ones((self.n_envs,1),device=self.device)
        self.steer_offsets        = self.get_randoms(self.n_envs,self.steering_offset)
        self.steer_delay          = self.get_randoms(self.n_envs,self.command_delay).to(torch.long)
        self.command_history      = torch.zeros((self.n_envs,10),device=self.device)

        self.envs_indexes_        = torch.arange(0, self.n_envs)
    
        self.viewer = virtual_screen_capture
        self.cfg = cfg


        #Gym params
        num_obs  = 8
        num_acts = 1

        buff_size = 5

        self.buffer_obs = torch.zeros((self.n_envs,buff_size,num_obs),device=rl_device)

        self.cfg["env"]["numObservations"]  = num_obs*buff_size
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
        self.dof_pos = self.state_dof.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.state_dof.view(self.num_envs, self.num_dof, 2)[..., 1]

                
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim) 

        self.root_reset_tensor  = self.root_tensor.clone().detach() # Copy root tensor
        self.root_reset_tensor[:, 7:13]  = 0 # Set all velocities to 0


        self.spawn_y = torch.zeros(self.num_envs,device=self.device)
        self.curent_perturbations = torch.zeros(self.n_envs,self.num_rgbd,3, device=self.device)

        self.reset_idx(torch.arange(0, self.n_envs).to(self.device).type(torch.long))




    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if(DEBUGUSETERRAIN):
            self._create_ground_plane()
        else:
            self._create_ground_plane_flat()
        self._create_envs()
        self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        self.terrain = Terrain()
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices =  self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = 0.0
        tm_params.transform.p.y = 0.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = 0.98
        tm_params.dynamic_friction = 0.98
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
    

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
        asset_file = "urdf/gogoro/urdf/scooter_V11.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = DEBUGFIXBASE
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.enable_gyroscopic_forces = True
        asset_options.disable_gravity = False
        asset_options.override_inertia = True # not sure what is best here, I don't think in the urdf the inertias are super good anyway
        asset_options.linear_damping = 0.01
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)


        # Get number of DOFs and DOF names
        self.num_dof   = self.gym.get_asset_dof_count(asset)
        self.dof_names = self.gym.get_asset_dof_names(asset)
        # Helper dictionary to map joint names to tensor ID
        self.dof_name_to_id = {k: v for k, v in zip(self.dof_names, np.arange(self.num_dof))}


        self.num_rgbd = self.gym.get_asset_rigid_body_count(asset)
        self.rgid_shape_to_id = {}
        for i in range(self.num_rgbd):
            if(self.gym.get_asset_rigid_body_shape_indices(asset)[i].count>0):
                self.rgid_shape_to_id[self.gym.get_asset_rigid_body_name(asset,i)] = self.gym.get_asset_rigid_body_shape_indices(asset)[i].start




        dof_props = self.gym.get_asset_dof_properties(asset)

        #because we lock the joints to their limits we need to do first put them inside
        self.thormang_pose = torch.zeros(self.n_envs,self.num_dof, device=self.device)




        # set up the env grid
        envs_per_row = self.num_envs#int(sqrt(self.num_envs))
        env_spacing  = 0.0
        env_lower    = gymapi.Vec3(-env_spacing, 0.0,        -env_spacing)
        env_upper    = gymapi.Vec3( env_spacing, env_spacing, env_spacing)

        self.envs    = []
        self.handles = []
        # create and populate the environments
        for i in range(self.num_envs):


            # Everything set to pos so they all stay in-place
            for d in range(self.num_dof):
                dof_props['driveMode'][d] = gymapi.DOF_MODE_NONE
                dof_props["damping"][d]   =  0.0
                dof_props["stiffness"][d] =  0.0
                dof_props["effort"][d]    =  0.0


            #locking the thormang joints
            for j_name in self.cfg["joints_pos"]:
                id_dof = self.dof_name_to_id[j_name]
                dof_props['driveMode'][id_dof] = gymapi.DOF_MODE_NONE
                dof_props["lower"][id_dof]   =  self.cfg["joints_pos"][j_name]
                dof_props["upper"][id_dof]   =  dof_props["lower"][id_dof]+0.0001
                self.thormang_pose[:,id_dof] =  dof_props["lower"][id_dof]+0.0001/2

            if(not CENTER_ROBOT):
                dof_props['driveMode'][self.dof_name_to_id['base_x']] = gymapi.DOF_MODE_NONE
                dof_props["lower"][self.dof_name_to_id['base_x']]     =  self.get_randoms(1,self.seat_offset_x_range).item()
                dof_props["upper"][self.dof_name_to_id['base_x']]     =  dof_props["lower"][self.dof_name_to_id['base_x']]+0.0001
                dof_props['driveMode'][self.dof_name_to_id['base_y']] = gymapi.DOF_MODE_NONE
                dof_props["lower"][self.dof_name_to_id['base_y']]     = self.get_randoms(1,self.seat_offset_y_range).item()
                dof_props["upper"][self.dof_name_to_id['base_y']]     = dof_props["lower"][self.dof_name_to_id['base_y']]+0.0001
                dof_props['driveMode'][self.dof_name_to_id['base_z']] = gymapi.DOF_MODE_NONE
                dof_props["lower"][self.dof_name_to_id['base_z']]     = self.get_randoms(1,self.seat_offset_z_range).item()
                dof_props["upper"][self.dof_name_to_id['base_z']]     = dof_props["lower"][self.dof_name_to_id['base_z']]+0.0001
            else:
                dof_props['driveMode'][self.dof_name_to_id['base_x']] = gymapi.DOF_MODE_NONE
                dof_props["lower"][self.dof_name_to_id['base_x']]     =  0.0
                dof_props["upper"][self.dof_name_to_id['base_x']]     =  dof_props["lower"][self.dof_name_to_id['base_x']]+0.0001
                dof_props['driveMode'][self.dof_name_to_id['base_y']] = gymapi.DOF_MODE_NONE
                dof_props["lower"][self.dof_name_to_id['base_y']]     = 0.0
                dof_props["upper"][self.dof_name_to_id['base_y']]     = dof_props["lower"][self.dof_name_to_id['base_y']]+0.0001
                dof_props['driveMode'][self.dof_name_to_id['base_z']] = gymapi.DOF_MODE_NONE
                dof_props["lower"][self.dof_name_to_id['base_z']]     = 0.0
                dof_props["upper"][self.dof_name_to_id['base_z']]     = dof_props["lower"][self.dof_name_to_id['base_z']]+0.0001

            
            self.thormang_pose[i,self.dof_name_to_id['base_x']] = dof_props["lower"][self.dof_name_to_id['base_x']]+0.0001/2
            self.thormang_pose[i,self.dof_name_to_id['base_y']] = dof_props["lower"][self.dof_name_to_id['base_y']]+0.0001/2
            self.thormang_pose[i,self.dof_name_to_id['base_z']] = dof_props["lower"][self.dof_name_to_id['base_z']]+0.0001/2



            # Set velocity mode on back wheel
            dof_props["driveMode"][self.dof_name_to_id['rear_wheel_joint']] = gymapi.DOF_MODE_VEL
            dof_props["stiffness"][self.dof_name_to_id['rear_wheel_joint']] = 0.0
            dof_props["damping"][self.dof_name_to_id['rear_wheel_joint']] = 100.0
            dof_props["effort"][self.dof_name_to_id['rear_wheel_joint']] = 130.0
            # Set position mode for steering joint
            if(RANDOM_DAMPING):
                dof_props["driveMode"][self.dof_name_to_id['steering_joint']] = gymapi.DOF_MODE_POS
                dof_props["stiffness"][self.dof_name_to_id['steering_joint']] =  100000.0
                dof_props["damping"][self.dof_name_to_id['steering_joint']]   =  self.get_randoms(1,self.steering_damping_range)
                dof_props["effort"][self.dof_name_to_id['steering_joint']]    =  100000.0
            else:
                dof_props["driveMode"][self.dof_name_to_id['steering_joint']] = gymapi.DOF_MODE_POS
                dof_props["stiffness"][self.dof_name_to_id['steering_joint']] =  100000.0
                dof_props["damping"][self.dof_name_to_id['steering_joint']]   =  5000.0
                dof_props["effort"][self.dof_name_to_id['steering_joint']]    =  100000.0
            #===================
            # dof_props["driveMode"][self.dof_name_to_id['bumper_front']] = gymapi.DOF_MODE_POS
            # dof_props["stiffness"][self.dof_name_to_id['bumper_front']] = 100000
            # dof_props["damping"][self.dof_name_to_id['bumper_front']] = 300
            # dof_props["effort"][self.dof_name_to_id['bumper_front']] = 10000
            # dof_props["friction"][self.dof_name_to_id['bumper_front']] = 0.1
            # dof_props["driveMode"][self.dof_name_to_id['bumper_back']] = dof_props["driveMode"][self.dof_name_to_id['bumper_front']]
            # dof_props["stiffness"][self.dof_name_to_id['bumper_back']] = dof_props["stiffness"][self.dof_name_to_id['bumper_front']]
            # dof_props["damping"][self.dof_name_to_id['bumper_back']]   = dof_props["damping"][self.dof_name_to_id['bumper_front']]
            # dof_props["effort"][self.dof_name_to_id['bumper_back']]    = dof_props["effort"][self.dof_name_to_id['bumper_front']]
            # dof_props["friction"][self.dof_name_to_id['bumper_back']]  = dof_props["friction"][self.dof_name_to_id['bumper_front']]


            ref_env      = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)

            start_pose   = gymapi.Transform()
            start_pose.p,start_pose.r,_ = self.generate_spawn_p_r()

            actor_handle = self.gym.create_actor(ref_env, asset, start_pose, "Gogoro", i, 1)
            actor_props  = self.gym.get_actor_rigid_shape_properties(ref_env, actor_handle)


            actor_props[self.rgid_shape_to_id["back"]].friction = 0.98
            actor_props[self.rgid_shape_to_id["back"]].rolling_friction = 0.0
            actor_props[self.rgid_shape_to_id["back"]].torsion_friction = 0.0

            actor_props[self.rgid_shape_to_id["front"]].friction         = 0.9
            actor_props[self.rgid_shape_to_id["front"]].rolling_friction = actor_props[self.rgid_shape_to_id["back"]].rolling_friction
            actor_props[self.rgid_shape_to_id["front"]].torsion_friction = actor_props[self.rgid_shape_to_id["back"]].torsion_friction


            self.gym.set_actor_rigid_shape_properties(ref_env, actor_handle, actor_props)
            self.gym.set_actor_dof_properties(ref_env, actor_handle, dof_props)

            self.envs.append(ref_env)
            self.handles.append(actor_handle)



    def generate_spawn_p_r(self):
        x_start = 0
        y_start = torch.rand(1,device=self.device)*100
        z_start = 0.01

        if(DEBUGUSETERRAIN):

            angle = torch.rand(1,device=self.device)*np.pi*2
            pixel_start_x = 5#int(self.terrain.Vx_shape/2) + int(math.cos(angle)*100)
            pixel_start_y = int(self.terrain.Vy_shape/2) + int((torch.rand(1,device=self.device)*2-1)*100)
            x_start = pixel_start_x*self.terrain.V_scale
            y_start = pixel_start_y*self.terrain.V_scale
            zone = 1
            z_start = self.terrain.heightsamples[pixel_start_x,pixel_start_y]*self.terrain.H_scale+0.01
            #z_start = torch.max(self.terrain.heightsamples[pixel_start_x-zone:pixel_start_x+zone,pixel_start_y-zone:pixel_start_y+zone])*self.terrain.H_scale+0.001

        # x_start = 0
        # y_start = 0

        initial_tilt    = 0.0
        initial_heading = 0.0
        return gymapi.Vec3(x_start, y_start, z_start), gymapi.Quat.from_euler_zyx(initial_tilt, 0, initial_heading), initial_heading




    def pre_physics_step(self, actions): 

        #raise Exception("TEST") 

        actions = torch.clamp(actions,-1,1)
        #self.curent_command = self.curent_command+actions*self.steering_sensitivity
        #self.curent_command = torch.clamp(self.curent_command,-1,1)
        self.curent_command = (actions[:,0]*self.max_steering)

        self.command_history[:,:-1]      = self.command_history[:,1:].clone()
        self.command_history[:, -1]      = self.curent_command
        applied_command = self.command_history[self.envs_indexes_,-self.steer_delay]

        dof_pos_command = torch.zeros(self.num_envs,self.num_dof, device=self.device)
        dof_pos_command[:,self.dof_name_to_id['steering_joint']]= applied_command+self.steer_offsets #+self.get_randoms(self.n_envs,self.steering_action_noise) #(self.curent_command[:,0]*self.max_steering)+self.get_randoms(self.n_envs,self.steering_action_noise)
        # dof_pos_command[:,self.dof_name_to_id['bumper_front']] = 0.0
        # dof_pos_command[:,self.dof_name_to_id['bumper_back']] = 0.0
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_pos_command))

        dof_speed_command = torch.zeros((self.num_envs,self.num_dof),device=self.device)
        dof_speed_command[:,self.dof_name_to_id['rear_wheel_joint']] = self.curent_speed
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_speed_command))


        if(DEBUG_YAW_RANDOM):
            yaw_command_change    = (self.progress_buf%self.yaw_freq_update == 0).nonzero(as_tuple=False).flatten()
            self.yaw_command[yaw_command_change] += self.get_randoms(self.n_envs,[-0.5,0.5])[yaw_command_change]


        else:
            target_points_X = self.root_positions[:,0]+5
            target_points_Y = self.spawn_y
            angle_to_target = torch.atan2(target_points_Y-self.root_positions[:,1],target_points_X-self.root_positions[:,0])
            self.yaw_command = angle_to_target




                    
    def post_physics_step(self):
        self.randomize_buf += 1
        self.progress_buf += 1

        speed_command_change = (self.progress_buf%self.speed_freq_update == 0).nonzero(as_tuple=False).flatten()
        self.curent_speed[speed_command_change] = 15.0#self.get_randoms(len(speed_command_change),self.speed_range)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        #Compute observations
        self.compute_observations()
        #Compute rewards
        self.compute_reward()


        self.gym.clear_lines(self.viewer)
        lines  = [self.root_positions[0,0].item(), self.root_positions[0,1].item(),self.root_positions[0,2].item()+1.3, 
                  self.root_positions[0,0].item()+torch.cos(self.buffer_obs[0,-1,1]).item()*3, self.root_positions[0,1].item()+torch.sin(self.buffer_obs[0,-1,1]).item()*3,self.root_positions[0,2].item()+1.3]
        colors = [1.0, 0.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[0],1,lines,colors)
        lines  = [self.root_positions[0,0].item(), self.root_positions[0,1].item(),self.root_positions[0,2].item()+1.3, 
                  self.root_positions[0,0].item()+torch.cos(self.yaw_command[0]).item()*3, self.root_positions[0,1].item()+torch.sin(self.yaw_command[0]).item()*3,self.root_positions[0,2].item()+1.3]
        colors = [0.0, 1.0, 0.0]
        self.gym.add_lines(self.viewer,self.envs[0],1,lines,colors)

        #print(self.rgid_shape_to_id)
        if(PUSH_ROBOT):
            need_update = (self.progress_buf+1)%50 == 0
            self.curent_perturbations[need_update,self.rgid_shape_to_id["head_p_link"],1] =  ((torch.rand(self.num_envs,device=self.device)*2-1)*10)[need_update]
            self.curent_perturbations[need_update,self.rgid_shape_to_id["head_p_link"],2] = -((torch.rand(self.num_envs,device=self.device))*10)[need_update]

            self.gym.apply_rigid_body_force_tensors(self.sim,gymtorch.unwrap_tensor(torch.flatten(self.curent_perturbations,end_dim=-2)),None)


            ypert = self.curent_perturbations[0,self.rgid_shape_to_id["head_p_link"],1]
            zpert = self.curent_perturbations[0,self.rgid_shape_to_id["head_p_link"],2]

            nrm =torch.abs(ypert)+torch.abs(zpert)
            ypert = (ypert/nrm).item()
            zpert = (zpert/nrm).item()
            
            # print(ypert)
            # print(zpert)
            # print("==")

            lines  = [  self.root_positions[0,0].item(), self.root_positions[0,1].item(),          self.root_positions[0,2].item()+1.3, 
                        self.root_positions[0,0].item(), self.root_positions[0,1].item()+ypert*3.0,self.root_positions[0,2].item()+1.3+zpert*3.0]
            colors = [0.0, 0.0, 1.0]

            self.gym.add_lines(self.viewer,self.envs[0],1,lines,colors)


            

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_gogoro_reward(
                                                                    root_poses         = self.root_positions,
                                                                    act_buff           = self.buffer_obs[:,:,-2], #buffer of actions
                                                                    curent_obs         = self.buffer_obs[:,-1],
                                                                    progress_buf       = self.progress_buf,
                                                                    max_episode_length = self.max_episode_length,
                                                                    )
        


    def compute_observations(self):
        # Refresh the tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        inv_start_rot = quat_conjugate(self.root_reset_tensor[:,3:7])

        current_steering = self.dof_pos[:,self.dof_name_to_id["steering_joint"]]

        obs_ = compute_gogoro_observations( 
                                            self.root_tensor.clone(),
                                            self.yaw_command,
                                            inv_start_rot,
                                            self.curent_command,
                                            current_steering,
                                            )
        
        obs_[:,0:2]  += self.get_randoms((self.n_envs,2),self.imu_filter_noise) #roll yaw
        obs_[:,2:4]  += self.get_randoms((self.n_envs,2),self.imu_noise)        #droll dyaw
        obs_[:,4]    += self.get_randoms_norm(self.n_envs,self.speed_sensor_noise) #speed
        obs_[:,4]     = torch.where(obs_[:,4]<0,0,obs_[:,4])

        obs_[:,5]    += self.get_randoms(self.n_envs,self.imu_filter_noise) #delta yaw
        obs_[:,6]    += 0.0  #curent_command

        obs_[:,7]    += self.get_randoms(self.n_envs,[-0.02,0.02]) #current_steering




        self.buffer_obs[:,:-1]      = self.buffer_obs[:,1:].clone()
        self.buffer_obs[:,-1]       = obs_
        self.obs_buf = self.buffer_obs.flatten(start_dim=1)


        #self.obs_buf[0] = self.obs_buf[1].clone()

        


        


    def get_randoms(self,shape,bounds):
        return bounds[0] + torch.rand(shape,device=self.device) * (bounds[1] - bounds[0])

    def get_randoms_norm(self,shape,mean_cov):
        return mean_cov[0] + torch.randn(shape,device=self.device) * mean_cov[1]


        
    def randomize(self,env_ids):

        self.apply_randomizations(self.randomization_params)
        _ids = env_ids
        n_id = _ids.shape[0]
        self.curent_speed[_ids]         = self.get_randoms(n_id,self.speed_range)
        self.steer_delay[_ids]          = self.get_randoms(n_id,self.command_delay).to(torch.long)
        self.steer_offsets[_ids]        = self.get_randoms(n_id,self.steering_offset)
        self.curent_perturbations[_ids] = torch.zeros(n_id,self.num_rgbd,3, device=self.device)




    def reset_idx(self, env_ids):
        nbresets = len(env_ids)
        env_ids_int32  = env_ids.to(dtype=torch.int32)
        _env_ids_int32 = gymtorch.unwrap_tensor(env_ids_int32) 

        self.randomize(env_ids_int32)


        for id in env_ids_int32:
            p,r, spawn_yaw = self.generate_spawn_p_r()
            self.root_reset_tensor[id,0] = p.x
            self.root_reset_tensor[id,1] = p.y
            self.root_reset_tensor[id,2] = p.z
            self.root_reset_tensor[id,3] = r.x
            self.root_reset_tensor[id,4] = r.y
            self.root_reset_tensor[id,5] = r.z
            self.root_reset_tensor[id,6] = r.w

            self.root_reset_tensor[id,7:13] = 0.0

            # start_vel = 2+(torch.rand(1,device = self.device))*3
            # x_vel = start_vel*math.cos(spawn_yaw)
            # y_vel = start_vel*math.sin(spawn_yaw)
            # self.root_reset_tensor[id,7]    = x_vel
            # self.root_reset_tensor[id,8]    = y_vel

            self.yaw_command[id] = spawn_yaw
            self.spawn_y[id] = p.y


        self.command_history[env_ids_int32,:] = 0.0

        assert self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_reset_tensor),_env_ids_int32, len(env_ids_int32))
            

        self.dof_pos[env_ids_int32] = self.thormang_pose[env_ids_int32]
        self.dof_vel[env_ids_int32] = 0.0
        assert self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.state_dof),_env_ids_int32, len(env_ids_int32))


        # # Reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.buffer_obs[env_ids] = 0.0
        self.curent_command[env_ids] = 0.0



#####################################################################
###=========================jit functions=========================###
#####################################################################



@torch.jit.script
def compute_gogoro_reward(root_poses,act_buff,curent_obs, progress_buf, max_episode_length):
    # # Reset if over iteration limit
    tilt    = curent_obs[:, 0]
    yaw_err = curent_obs[:, 5]

    reward1 =  1-(8*tilt)**4
    reward2 =  1-(2*yaw_err)**2 #max = 0 min = -50*pi
    reward3 = -torch.mean(torch.diff(act_buff,dim=1)**2,dim=1)*30

    #print("=====")
    #print("act_buff=",act_buff[0])
    #print("diff    =",torch.diff(act_buff,dim=1)[0])
    #print("yaw_rwd =",reward2[0].item())
    #print("act_rwd =",reward3[0].item())
    reward = reward1+reward2 +reward3

    reset  = torch.where(progress_buf > max_episode_length, 1, 0)
    reset  = torch.where(torch.abs(tilt)    > 0.11,1,    reset)
    # reset  = torch.where(torch.abs(yaw_err) > 0.8, 1,    reset)
    #reward = torch.where(torch.abs(tilt)    > 0.11,reward+max_episode_length*-100, reward)
    # reward = torch.where(torch.abs(yaw_err) > 0.8, reward+max_episode_length*-100 , reward)
    # reward = torch.where(progress_buf > max_episode_length, 100, reward)

    return reward, reset


@torch.jit.script
def shortest_angle_distance(yaw1, yaw2):
    return (yaw2 - yaw1 + torch.pi)%(2*torch.pi) - torch.pi


@torch.jit.script
def compute_gogoro_observations(root_state,desired_yaw,inv_start_rot,curent_command, current_steering):
    
    root_orientations = root_state[:, 3:7]
    root_linvel       = root_state[:, 7:10]
    root_angvel       = root_state[:, 10:13]

    roll, _, yaw = get_euler_xyz(root_orientations)#get_euler_xyz(torso_quat)

    torso_quat   = quat_mul(root_orientations,inv_start_rot)
    base_ang_vel = quat_rotate_inverse(torso_quat, root_angvel)


    d_roll  = base_ang_vel[:,0]
    d_yaw   = base_ang_vel[:,2]

    roll = torch.where(roll>np.pi,roll-np.pi*2,roll)
    yaw  = torch.where(yaw>np.pi,yaw-np.pi*2,yaw)

    delta_yaw = shortest_angle_distance(yaw, desired_yaw)  

    speed = torch.sum(torch.abs(root_linvel[:,:2]),dim=1)
    

    obs = torch.cat([
                        roll.unsqueeze(-1),
                        yaw.unsqueeze(-1),
                        d_roll.unsqueeze(-1),
                        d_yaw.unsqueeze(-1),
                        speed.unsqueeze(-1),
                        delta_yaw.unsqueeze(-1),
                        curent_command.unsqueeze(-1),
                        current_steering.unsqueeze(-1),
                        ], dim=1
                    )

    return obs 




        








from isaacgym.terrain_utils import *
import matplotlib.pyplot as plt
import math
class Terrain:
    def __init__(self) -> None:
        
        self.Vx_shape = 256 
        self.Vy_shape = 256 

        self.V_scale = 0.5
        self.H_scale = 1.0

        self.Vx_size_m = self.V_scale*self.Vx_shape
        self.Vy_size_m = self.V_scale*self.Vy_shape


        self.heightsamples = self.rand_perlin_2d_octaves((self.Vx_shape, self.Vy_shape), (1, 4), 2)

        print(self.heightsamples.shape)


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
