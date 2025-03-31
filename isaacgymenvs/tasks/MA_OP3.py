import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from .base.multi_vec_task import MA_VecTask
from typing import Tuple, Dict

class MA_OP3(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales['heading_scale'] = self.cfg["env"]["learn"]["headingScale"]
        self.rew_scales['up_scale'] = self.cfg["env"]["learn"]["upScale"]
        self.rew_scales['air_time'] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales['syns_hip'] = self.cfg["env"]["learn"]["syncronizeHipRewardScale"]
        self.rew_scales['no_fly'] = self.cfg["env"]["learn"]["noflyRewardScale"]
        self.rew_scales['stand_scale'] = self.cfg["env"]["learn"]["standRewardScale"]
        self.rew_scales['action_rate'] = self.cfg["env"]["learn"]["actionRateRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        self.command_heading_range = self.cfg["env"]["randomCommandVelocityRanges"]["heading"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["restitution"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 1
        self.cfg["env"]["numActions"] = 1

        # other
        self.dt = self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.op3_body_colors = [gymapi.Vec3(*rgb_arr) for rgb_arr in self.cfg["env"]["color"]]
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, 
                         graphics_device_id=graphics_device_id, headless=headless, 
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        table_per_env = 1 # 0 : no table 1: table
        actors_per_env =  table_per_env + self.num_agents
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.0, 1.0, 0.8)
            cam_target = gymapi.Vec3(0.0, 0, 0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, actors_per_env, 13)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, :, 7:13] = 0
        #Table has reverse direction on the plane this table is move to X direction
        self.goal_pos = to_torch([0, -10, 0], device=self.device).repeat((self.num_envs, 1))
        
        self.table_state = self.root_states[:, 2]
        self.table_positions = self.table_state[:, 0:3]
        self.table_linvels  =  self.table_state[:,7:10]

        self.table_potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_table_potentials = self.table_potentials.clone()

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_agents, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_agents, self.num_dof, 2)[..., 1]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)     
        
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs,-1, 3)# shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs,self.num_agents,self.num_dof)

        self.commands = torch.zeros(self.num_envs,3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_x = self.commands.view(self.num_envs,3)[...,0]
        self.commands_y = self.commands.view(self.num_envs,3)[..., 1]
        self.commands_yaw = self.commands.view(self.num_envs,3)[..., 2]

        self.op3_init_xy = torch.tensor(self.base_init_state[0:2], device=self.device)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, :,i] = angle
        
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs,self.num_agents ,1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs,self.num_agents,len(self.feet_indices[0]), dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs,self.num_agents,len(self.feet_indices[0]), dtype=torch.bool, device=self.device, requires_grad=False)
        # self.reward_step = torch.zeros((self.num_envs), device=self.device)
        self.last_actions = self.actions.clone()

        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs,self.num_agents + 1, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs,self.num_agents + 1, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()        
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs,self.num_agents, 1))
        self.targets = to_torch([0, 10, 0], device=self.device).repeat((self.num_envs, 1))
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs, self.num_agents)
        self.prev_potentials = self.potentials.clone()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
      
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        table_options = gymapi.AssetOptions()
        table_options.armature = 0.01
        table_options.density = 2.5
        table_options.fix_base_link = False # ! importop3
        table_options.use_mesh_materials = True
        table_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        table_options.override_inertia = True
        table_options.override_com = True
        # table_options.vhacd_enabled = True
        # table_options.vhacd_params = gymapi.VhacdParams()
        # table_options.vhacd_params.resolution = 100
        table_file = "urdf/TableV2/urdf/TableV4.urdf" #/home/ntnu/IsaacGymEnvs/assets/urdf/table/urdf/table.urdf
        table_asset = self.gym.load_asset(self.sim, asset_root, table_file,table_options)
        
        asset_file = "op3_description/urdf/robotis_op3v3.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.fix_base_link = False
        asset_options.density = 5
        asset_options.angular_damping = 0.4
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 10
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 10
        # asset_options.vhacd_params.max_convex_hulls = 2
        # asset_options.vhacd_params.max_num_vertices_per_ch = 2
        # asset_options.convex_decomposition_from_submeshes = True
        

        op3_assets = []
        for _ in range(self.num_agents):
            op3_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            op3_assets.append(op3_asset)
        dof_props = self.gym.get_asset_dof_properties(op3_assets[0])
        
        self.num_dof = self.gym.get_asset_dof_count(op3_assets[0])
        self.num_bodies = self.gym.get_asset_rigid_body_count(op3_assets[0])
        self.num_bodies_table = self.gym.get_asset_rigid_body_count(table_asset)
        body_names = self.gym.get_asset_rigid_body_names(op3_assets[0])
        self.dof_names = self.gym.get_asset_dof_names(op3_assets[0])

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(-0.31, 0.0, 0.27)
        start_pose.r = gymapi.Quat.from_euler_zyx(*[0.0, 0.0, 0.0])

        start_pose_a = gymapi.Transform()
        start_pose_a.p = gymapi.Vec3(0.30, 0.0, 0.27)
        start_pose_a.r = gymapi.Quat.from_euler_zyx(*[0.0, 0.0, 3.14])

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0, 0.30)
        table_pose.r = gymapi.Quat.from_euler_zyx(*[0.0, 0.0, 1.57])

        # Create a list of tensors for start_pose and start_pose_a
        pose_list = [start_pose, start_pose_a]
        self.start_rotation = torch.tensor([pose_list[0].r.x, 
                                            pose_list[0].r.y, 
                                            pose_list[0].r.z, 
                                            pose_list[0].r.w], device=self.device)

        extremity_name = "SHANK" if asset_options.collapse_fixed_joints else "ank"
        feet_names = [s for s in body_names if extremity_name in s]
        self.feet_indices = torch.zeros((self.num_agents, len(feet_names)), dtype=torch.long, device=self.device, requires_grad=False)
        knee_names = [s for s in body_names if "knee" in s]
        self.knee_indices = torch.zeros((self.num_agents, len(knee_names)), dtype=torch.long, device=self.device, requires_grad=False)
        gripper_names = [s for s in body_names if "gr" in s]
        self.gripper_indices = torch.zeros((self.num_agents, len(gripper_names)), dtype=torch.long, device=self.device, requires_grad=False)
        elbow_names = [s for s in body_names if "elbow" in s]
        self.elbow_indices = torch.zeros((self.num_agents, len(elbow_names)), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index_= 0

        dof_props = self.gym.get_asset_dof_properties(op3_assets[0])
        dof_props_coop = self.gym.get_asset_dof_properties(op3_assets[1])
        
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT #self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"] 
            dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

            dof_props_coop['driveMode'][i] = gymapi.DOF_MODE_EFFORT #self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"] 
            dof_props_coop['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props_coop['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)     

        self.op3_handles = []
        self.actor_indices = []
        self.object_indices = []
        self.envs = []
 
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            op3_idx = []
            for j in range(self.num_agents):
                op3_handle = self.gym.create_actor(env_ptr, op3_assets[j], pose_list[j], "op3_" + str(j), i, 1, 1)
                op3_idx.append(self.gym.get_actor_index(env_ptr, op3_handle, gymapi.DOMAIN_SIM))
                self.gym.set_actor_dof_properties(env_ptr, op3_handle, dof_props)
                self.gym.enable_actor_dof_force_sensors(env_ptr, op3_handle)

                for k in range(self.num_bodies):
                    self.gym.set_rigid_body_color(
                        env_ptr, op3_handle, k, gymapi.MESH_VISUAL, self.op3_body_colors[j])

                self.op3_handles.append(op3_handle)  # Move this line inside the inner loop
            self.actor_indices.append(op3_idx)
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 2, 1)
            self.object_indices.append(self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM))

            self.envs.append(env_ptr)
        self.op3_handles = list(set(self.op3_handles))
        # print(self.op3_handles)
        self.actor_indices = to_torch(self.actor_indices, dtype=torch.long, device=self.device)
        # self.actor_indices_coop = to_torch(self.actor_indices_coop, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        
        for i in range(len(feet_names)):
            self.feet_indices[0, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], feet_names[i])
            self.feet_indices[1, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[1], feet_names[i])
        for i in range(len(knee_names)):
            self.knee_indices[0, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], knee_names[i])
            self.knee_indices[1, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[1], knee_names[i])
        for i in range(len(gripper_names)):
            self.gripper_indices[0, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], gripper_names[i])
            self.gripper_indices[1, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[1], gripper_names[i])
        for i in range(len(elbow_names)):
            self.elbow_indices[0, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], elbow_names[i])
            self.elbow_indices[1, i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[1], elbow_names[i])

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[0], "base-frame-link")
        self.base_index_coop = self.gym.find_actor_rigid_body_handle(self.envs[0], self.op3_handles[1], "base-frame-link")

    def pre_physics_step(self, actions):
        pass
        

    def post_physics_step(self):

        self.obs_buf   = torch.zeros((self.num_envs,1),device=self.device)
        self.rew_buf[:]   = 0
        self.reset_buf[:] = 0

    def _reward_air_time(self,agent_idx):
        feet_agent = self.feet_indices[agent_idx]
        contact = self.contact_forces[:, feet_agent, 2] > 1.1
        contact_filt = torch.logical_or(contact, self.last_contacts[:,agent_idx]) 
        self.last_contacts[:,agent_idx] = contact
        first_contact = (self.feet_air_time[:,agent_idx] > 0.) * contact_filt
        self.feet_air_time[:,agent_idx] += self.dt
        rew_airTime = torch.sum((self.feet_air_time[:,agent_idx] - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        self.feet_air_time[:,agent_idx] *= ~contact_filt
        return rew_airTime

    def _reward_feet_contact_forces(self,agent_idx):
        # penalize high contact forces
        feet_agent = self.feet_indices[agent_idx]
        return torch.sum((torch.norm(self.contact_forces[:,feet_agent, :], dim=-1) -  450).clip(min=0.), dim=1)
    
    def _reward_no_fly(self,agent_idx):
        feet_agent = self.feet_indices[agent_idx]
        contacts = self.contact_forces[:, feet_agent, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_gripper_contact_forces(self,agent_idx):
        gripper_agent = self.gripper_indices[agent_idx]
        return torch.sum((torch.norm(self.contact_forces[:,gripper_agent,:], dim=-1) -  400).clip(min=0.), dim=1)

    def _reward_gripper_contact_hold(self, agent_idx):
        grippers_agent_left  = self.gripper_indices[agent_idx,0]
        grippers_agent_right = self.gripper_indices[agent_idx,1]
        # print(self.gripper_indices.shape)
        left_contacts  = self.contact_forces[:, grippers_agent_left, 0] > 0.1
        right_contacts = self.contact_forces[:, grippers_agent_right, 0] > 0.1
        # print(left_contacts.shape)
        successful_left_grip = (left_contacts.sum() == 1)
        successful_right_grip = (right_contacts.sum() == 1)
        successful_both_grips = successful_left_grip and successful_right_grip
        successful_both_grips_tensor = torch.tensor(successful_both_grips).float()
        
        return successful_both_grips

    def compute_reward(self, actions):
        for i in range(self.num_agents):
            reward_airtime = self._reward_air_time(i) * self.rew_scales["air_time"]
            reward_step    = self._reward_feet_contact_forces(i)
            reward_no_fly  = self._reward_no_fly(i) * self.rew_scales["no_fly"]
            #reward gripper how to gripper
            reward_gripper = self._reward_gripper_contact_forces(i) * 0.5
            reward_gripper_hold = self._reward_gripper_contact_hold(i)


            self.rew_buf[:,i], self.reset_buf[:] = compute_op3_reward(
                    self.root_states[:,i],
                    self.commands,
                    self.prev_torques[:,i],
                    self.torques[:,i],
                    self.progress_buf,
                    self.rew_scales,
                    self.max_episode_length,
                    self.targets,
                    self.heading_vec[:,i],
                    self.up_vec[:,i],
                    self.inv_start_rot[:,i],
                    self.reset_buf,
                    self.actions[:,i],
                    self.last_actions[:,i],
                    self.dof_pos[:,i],
                    self.default_dof_pos[:,i],
                    self.potentials[:,i],
                    self.prev_potentials[:,i],
                    reward_airtime,
                    reward_step,
                    reward_no_fly,
                    reward_gripper,
                    reward_gripper_hold,
                    self.gravity_vec[:, i],
                    self.table_state,
                )
        self.rew_buf[:], self.reset_buf[:] = compute_objective_reward(
                    self.rew_buf[:], 
                    self.reset_buf[:], 
                    self.targets, 
                    self.table_positions,
                    self.root_states[:,self.num_agents], 
                    self.initial_root_states[:, self.num_agents, 0:3],
                    self.table_potentials,
                    self.prev_table_potentials,
                    self.basis_vec0[:,2],
                    self.basis_vec1[:,2],
                    self.num_envs,
            )
            
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.prev_obj_potentials = self.table_potentials.clone()
        to_target_goal = self.targets - self.table_positions
        self.prev_table_potentials = -torch.norm(to_target_goal, p=2, dim=-1) / self.dt

        for i in range(self.num_agents):
            self.obs_buf[:, i],self.potentials[:, i],self.prev_potentials[:,i] = compute_op3_observations(self.root_states[:, i],
                                                                                                          self.targets,
                                                                                                          self.dof_pos[:, i],
                                                                                                          self.default_dof_pos[:, i],
                                                                                                          self.dof_vel[:, i],
                                                                                                          self.gravity_vec[:, i],
                                                                                                          self.actions[:, i],
                                                                                                          self.table_state, 
                                                                                                          # scales, i
                                                                                                          self.lin_vel_scale,
                                                                                                          self.ang_vel_scale,
                                                                                                          self.dof_pos_scale,
                                                                                                          self.dof_vel_scale,
                                                                                                          self.dt,
                                                                                                          self.potentials[:, i],
                                                                                                          self.goal_pos, #Arah sebaliknya robot ke titik tujuan
                                                                                                        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch.zeros((len(env_ids), self.num_agents, 2), device=self.device)  # Temporarily generating for 2 DOFs
        velocities = torch.zeros((len(env_ids), self.num_agents, self.num_dof), device=self.device)

        # Initialize a positions offset tensor filled with zeros
        positions_offset_specific = torch.zeros((len(env_ids), self.num_agents, self.num_dof), device=self.device)

        # Correcting indices based on 0-based indexing (Python) and applying the random values to specific DOFs
        positions_offset_specific[:, :, 11] = positions_offset[:, :, 0]  # For DOF 11 (0-based index 11)
        positions_offset_specific[:, :, 21] = positions_offset[:, :, 1]  # For DOF 21 (1-based index 21)

        # Applying the adjusted positions and velocities
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] + positions_offset_specific
        self.dof_vel[env_ids] = velocities


        op3_indices = self.actor_indices[env_ids].to(dtype=torch.int32)
        all_actor_indices = torch.cat((self.actor_indices, self.object_indices.unsqueeze(1)), dim=1)
        all_actor_indices = all_actor_indices[env_ids].to(dtype=torch.int32)
        all_actor_indices = all_actor_indices.flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), 
                                                gymtorch.unwrap_tensor(op3_indices), 
                                                len(op3_indices))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(all_actor_indices), 
                                                     len(all_actor_indices))

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        to_target = self.goal_pos[env_ids].unsqueeze(1) - self.initial_root_states[env_ids, 0:self.num_agents, 0:3]
        to_target[:,:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        to_target_goal = self.targets[env_ids] - self.initial_root_states[env_ids, 2, 0:3]
        to_target_goal[:, 2] = 0.0
        self.prev_table_potentials[env_ids] = -torch.norm(to_target_goal, p=2, dim=1) / self.dt
        self.table_potentials[env_ids] = self.prev_table_potentials[env_ids].clone()

        self.last_actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
   

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_objective_reward(
    reward_agent,
    reset_buf,
    target_pos,
    object_pos,
    table_root,
    initial_pos,
    potentials,
    prev_potentials,
    vec0,
    vec1,
    num_envs,
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor,Tensor,Tensor,int) -> Tuple[Tensor, Tensor]
    initial_goal_dist = torch.norm(target_pos - initial_pos, p=2, dim=-1)
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

    table_position = table_root[:,0:3]
    table_quat     = table_root[:,3:7]
    table_linvels  = table_root[:,7:10]
    table_height = table_position[:, 2] 

    to_target = target_pos - table_position
    to_target[:, 2] = 0
    target_dirs = normalize(to_target)

    heading_vec = get_basis_vector(table_quat, vec0).view(num_envs, 3)
    heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    heading_weight_tensor = torch.ones_like(heading_proj) 
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_proj / 0.8)

    up_vec = get_basis_vector(table_quat, vec1).view(num_envs, 3)
    up_proj = up_vec[:, 2]

    # Aligning up axis of OP3 and environment
    rew_up = torch.zeros_like(heading_reward)
    rew_up = torch.where(up_proj > 0.98, rew_up + 0.1, rew_up)
    rew_up = torch.cat((rew_up.unsqueeze(1),rew_up.unsqueeze(1)), dim=1)

    rew_base_height = torch.square(table_position[:, 2] - 0.29) * -0.001
    # print(table_position[:, 2])
    rew_base_height = torch.cat((rew_base_height.unsqueeze(1),rew_base_height.unsqueeze(1)), dim=1)

    dist_rew = (potentials - prev_potentials)*5
    dist_rew = torch.cat((dist_rew.unsqueeze(1),dist_rew.unsqueeze(1)), dim=1)
    rew_buf = reward_agent + dist_rew + rew_up + rew_base_height

    reset_buf = torch.where(up_proj < 0.90, torch.ones_like(reset_buf) , reset_buf)
    reset_buf = torch.where(table_height < 0.25, torch.ones_like(reset_buf) , reset_buf)

    return rew_buf, reset_buf

@torch.jit.script
def compute_op3_reward(
    root_states,
    commands,
    prev_torques,
    torques,
    episode_lengths,
    rew_scales: Dict[str, float],
    max_episode_length: int,
    targets,
    vec0,
    vec1,
    inv_start_rot,
    reset_buf,
    actions,
    last_actions,
    dof_pos,
    default_dof_pos,
    potentials,
    prev_potentials,
    rew_airtime,
    rew_step,
    rew_no_fly,
    rew_gripper,
    rew_gripper_hold,
    gravity_vec,
    root_table,
    ):  
    # Prepare quantities (TODO: return from obs?)
    quat = root_states[:, 3:7]
    ang_vel = quat_rotate_inverse(quat, root_states[:, 10:13])

    position = root_states[:, 0:3]
    rotation = root_states[:, 3:7]

    z_position = root_states[:, 2]
    op3_z_position = torch.clamp((z_position - 0.0) / (0.27 - 0.0), 0.0, 1.0)

    #Get vector of heading and upright
    to_target = targets - position
    to_target[:, 2] = 0
    target_dirs = normalize(to_target)

    num_envs = position.shape[0]
    torso_quat = quat_mul(rotation, inv_start_rot)

    heading_vec = get_basis_vector(torso_quat, vec0).view(num_envs, 3)
    heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    # Reward from direction headed
    heading_weight_tensor = torch.ones_like(heading_proj) * rew_scales["heading_scale"]
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, rew_scales["heading_scale"] * heading_proj / 0.8)
    up_vec = get_basis_vector(torso_quat, vec1).view(num_envs, 3)
    up_proj = up_vec[:, 2]

    # Aligning up axis of ant and environment
    rew_up = torch.zeros_like(heading_reward)
    rew_up = torch.where(up_proj > 0.95, rew_up + rew_scales['up_scale'], rew_up)

    # reward for duration of staying alive
    to_target = targets - position
    to_target[:, 2] = 0.0
    alive_reward = torch.ones_like(potentials) * 2
    progress_reward = (potentials - prev_potentials) * 5

    # projected gravity reward
    projected_gravity = quat_rotate_inverse(quat, gravity_vec)
    rew_orient = torch.sum(torch.square(projected_gravity[:, :2]), dim=1) * -0
    
    # Pinalty for Torque
    rew_torque = torch.sum(torch.abs(prev_torques - torques), dim=1) * rew_scales["torque"]
    
    # Pinalty for leg of the robot
    leg_position_pinalty = torch.sum(torch.abs(dof_pos[:, [2,3,4,5,6,7,11,12,13,14,15,16,17,21]]-\
                                               default_dof_pos[:,[2,3,4,5,6,7,11,12,13,14,15,16,17,21]]),dim=1)
    rew_syns = leg_position_pinalty * rew_scales["syns_hip"]
    
    # Action rate (reward)
    rew_action_rate = torch.sum(torch.square(last_actions - actions), dim=1) * rew_scales["action_rate"]

    # Pinalty still stand 
    rew_stand_still = torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1) * (torch.norm(commands[:, :2], dim=1) < 0.1) * rew_scales["stand_scale"]
    
    ang_vel_error = torch.square(targets[:, 2] - ang_vel[:, 2])
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.1) * 0.1

    #keep close with gripper
    # pos_gripleft  = dof_pos[0,11]
    # pos_gripright = dof_pos[:,12]

    # print(pos_gripleft)

    #keep close with table
    max_distance = 0.32
    scale_reward = 0.5
    agent_distance_to_table  = torch.norm(position - root_table[:,:3],p=2,dim=-1)
    rew_distobject = torch.exp(-agent_distance_to_table / max_distance) * scale_reward

    #Total Reward 
    total_reward = progress_reward + alive_reward + rew_torque + rew_up  + rew_airtime + rew_ang_vel_z + rew_step +\
                   rew_no_fly +rew_stand_still + rew_action_rate + rew_syns + rew_orient + rew_distobject  + rew_gripper_hold
    total_reward = torch.clip(total_reward, 0., None)

    # Out of Bound
    reset_buf[episode_lengths >= max_episode_length] = 1
    reset_buf = torch.where(up_proj < 0.90, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(agent_distance_to_table > 0.40,torch.ones_like(reset_buf), reset_buf)
    reset_buf[episode_lengths == 0] = 0
    # Alive reward

    return total_reward.detach(), reset_buf

@torch.jit.script
def compute_op3_observations(root_states,
                            target,
                            dof_pos,
                            default_dof_pos,
                            dof_vel,
                            gravity_vec,
                            actions,
                            object_state,
                            lin_vel_scale,
                            ang_vel_scale,
                            dof_pos_scale,
                            dof_vel_scale,
                            dt,
                            potentials,
                            goal_pos,
                                ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,float, float, float, float, float, Tensor,Tensor) -> Tuple[Tensor, Tensor, Tensor]
    torso_position = root_states[:, 0:3]
    to_target = goal_pos - torso_position
    
    to_target[:, 2] = 0.0

    object_pose = object_state[:, 0:7]
    object_linvel = object_state[:, 7:10]
    object_angvel = object_state[:, 10:13]

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions,
                     object_pose,
                     object_linvel,
                     target,
                     ), dim=-1)
    
    return obs,potentials,prev_potentials_new

@torch.jit.script
def expand_env_ids(env_ids, n_agents):
    # type: (Tensor, int) -> Tensor
    device = env_ids.device
    # print(f'nanget:{n_agents}')
    agent_env_ids = torch.zeros((n_agents * len(env_ids)), device=device, dtype=torch.long)
    for idx in range(n_agents):
        agent_env_ids[idx::n_agents] = env_ids * n_agents + idx
    return agent_env_ids