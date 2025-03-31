#python3 export.py task=Gogoro checkpoint="/home/erc/RL_NVIDIA/IsaacGymEnvs/isaacgymenvs/runs/Gogoro/nn/Gogoro.pth"


import datetime
import isaacgym
import torch
import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed
## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner,_restore,_override_sigma
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    assert cfg.checkpoint

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
            monitor_gym=True,
        )

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    #runner.reset()

    # dump config dict
    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


    player = runner.create_player()
    player.restore(cfg.checkpoint)


    class ModelWrapper(torch.nn.Module):
        '''
        Main idea is to ignore outputs which we don't need from model
        '''
        def __init__(self, model):
            torch.nn.Module.__init__(self)
            self._model = model
            
            
            
        def forward(self,obs):

            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : obs,
                'rnn_states' : None
            }
            res_dict = self._model(input_dict)

            mu = res_dict['mus']
            current_action = mu

            return torch.clamp(current_action, -1.0, 1.0)



    import rl_games.algos_torch.flatten as flatten
    n_envs = 1
    inputs = torch.zeros((n_envs,) + player.obs_shape).to(player.device),

    with torch.no_grad():
        adapter = flatten.TracingAdapter(ModelWrapper(player.model), inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        flattened_outputs = traced(*adapter.flattened_inputs)
        print(flattened_outputs)


    torch.onnx.export(traced, *adapter.flattened_inputs, "./exports/test.onnx", verbose=True, input_names=['obs'], output_names=['actions'])


    print("generating test set for extra checking ...")
    import onnx
    import onnxruntime as ort
    import numpy as np

    onnx_model = onnx.load("./exports/test.onnx")
    onnx.checker.check_model(onnx_model)
    loaded_model = ort.InferenceSession("./exports/test.onnx")

    fake_obs_list = []
    output_list   = []

    print("player.obs_shape=",player.obs_shape)
    for i in range(100):
        fake_obs = (np.random.rand(n_envs ,player.obs_shape[0])*2-1) #
        outputs = loaded_model.run(
                                            None,
                                            {"obs": fake_obs.astype(np.float32)},
                                        )

        fake_obs_list.append(fake_obs)
        output_list.append(outputs)

    d = {"obs":fake_obs_list,"outputs":output_list}
    
    np.save("./exports/test.onnx.npy",d)

if __name__ == "__main__":
    launch_rlg_hydra()
