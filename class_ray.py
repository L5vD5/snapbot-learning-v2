import numpy as np
import ray
import torch
from class_grp import GaussianRandomPathClass, scaleup_traj, get_anchors_from_traj

@ray.remote(num_cpus=2)
class RayRolloutWorkerClass:
    def __init__(self, env, device=None, worker_id=1) -> None:
        self.env = env(render_mode=None)
        self.device = device

        print("init worker {}".format(worker_id))
        pass

    def generate_trajectory(self, DLPG, lbtw, dur_sec, hyp_prior, hyp_posterior, GRPPrior, GRPPosterior, prior_prob, start_epoch, n_anchor, t_anchor, traj_secs):
        exploration_coin = np.random.rand()
        if self.env.condition is not None:
            condition_coin = np.random.uniform(0, 1)
            if condition_coin >= 0.7:
                c = np.array([1,0,0])
            elif condition_coin >= 0.3:
                c = np.array([0,1,0])
            else:
                c = np.array([0,0,1])
        else:
            c = np.array([0,1,0])
        if (exploration_coin < prior_prob) or (start_epoch < 1):
            GRPPrior.set_prior(n_data_prior=4, dim=self.env.adim, dur_sec=dur_sec, HZ=self.env.hz, hyp=hyp_prior)
            traj_joints, traj_secs = GRPPrior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0) 
            traj_joints_deg = scaleup_traj(self.env, traj_joints, DO_SQUASH=True, squash_margin=5)
        else:
            x_anchor = DLPG.sample_x(c=torch.FloatTensor(c).reshape(1,-1).to(self.device), n_sample=1).reshape(n_anchor, self.env.adim)
            x_anchor[-1,:] = x_anchor[0,:]
            GRPPosterior.set_posterior(t_anchor, x_anchor,lbtw=lbtw, t_test=traj_secs, hyp=hyp_posterior, APPLY_EPSRU=True, t_eps=0.025)
            traj_joints, _ = GRPPosterior.sample_one_traj(rand_type='Uniform', ORG_PERTURB=True, perturb_gain=0.0) 
            traj_joints_deg = scaleup_traj(self.env, np.array(traj_joints), DO_SQUASH=True, squash_margin=5)
        t_anchor, x_anchor = get_anchors_from_traj(traj_secs, traj_joints, n_anchor=n_anchor)

        return {'x_anchor':x_anchor, 'c': c, 'traj_joints_deg': traj_joints_deg}

    def rollout(self, PID, traj_scale, n_traj_repeat, RENDER=False, PLOT=False):
        self.env.reset()
        PID.reset()
        L       = traj_scale.shape[0]
        secs    = np.zeros(shape=(L*n_traj_repeat))
        xy_degs = np.zeros(shape=(L*n_traj_repeat, 3))
        x_prev  = self.env.get_body_com("torso")[0]
        forward_rewards, left_rewards, right_rewards = [], [], []
        cnt     = 0
        for traj_idx in range(n_traj_repeat):
            for tick in range(L):
                PID.update(x_trgt=traj_scale[tick,:], t_curr=self.env.get_time(), x_curr=self.env.get_joint_pos_deg())
                _, every_reward, done, rwd_detail = self.env.step(PID.out())
                if done:
                    break
                if self.env.condition is not None:
                    forward_rewards.append(every_reward[0])
                    left_rewards.append(every_reward[1])
                    right_rewards.append(every_reward[2])
                else:
                    forward_rewards.append(every_reward)
                secs[cnt] = self.env.get_time()
                xy_degs[cnt, :] = np.concatenate((self.env.get_body_com("torso")[:2],[self.env.get_heading()]))
                cnt = cnt + 1
                if RENDER:
                    self.env.render()
        x_final = self.env.get_body_com("torso")[0]
        x_diff  = x_final - x_prev

        return {'secs': secs, 'xy_degs':xy_degs, 'forward_rewards': forward_rewards, 'x_diff': x_diff}

# Not Implemented
if __name__ == "__main__":
    pass
    # ray.init()
    # assert ray.is_initialized()

    # rollout_list = [ray_rollout.remote(print, 'i') for i in range(10)]

    # a = ray.get(rollout_list)

    # ray.shutdown()
    # assert not ray.is_initialized()

