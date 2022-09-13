from re import S
import cv2,math,os,warnings,time,random
warnings.filterwarnings("ignore") # ignore warnings

# Plot
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['animation.embed_limit'] = 2**128
from matplotlib import animation
from IPython.display import display, HTML

# Torch
import torch
import torch.nn as nn  
import torch.nn.functional as F

# Computation
import numpy as np
from scipy.spatial.distance import cdist

# Mujoco and gym
import mujoco_py
from gym.envs.mujoco import mujoco_env
from gym import utils

# DPP
from dpp import ikdpp

def quaternion_to_euler_angle(w, x, y, z):
    y_sqr = y*y

    t_0 = +2.0 * (w*x + y*z)
    t_1 = +1.0 - 2.0 * (x*x + y_sqr)
    X = math.degrees(math.atan2(t_0, t_1))
	
    t_2 = +2.0 * (w*y - z*x)
    t_2 = +1.0 if t_2 > +1.0 else t_2
    t_2 = -1.0 if t_2 < -1.0 else t_2
    Y = math.degrees(math.asin(t_2))
	
    t_3 = +2.0 * (w*z + x*y)
    t_4 = +1.0 - 2.0 * (y_sqr + z*z)
    Z = math.degrees(math.atan2(t_3, t_4))
	
    return X, Y, Z

# Snapbot Class
class SnapbotEnvClass(mujoco_env.MujocoEnv,utils.EzPickle):
    def __init__(self,
                 name        = 'SB',
                 xml_path    = '../xml/robot_4_1245.xml',
                 frame_skip  = 5,
                 render_mode = 'rgb_array',
                 render_w    = 1500,
                 render_h    = 1000,
                 render_res  = 200
                ):
        self.name       = name
        self.xml_path   = os.path.abspath(xml_path)
        self.frame_skip = frame_skip

        # Set joint position limit in [deg]
        self.joint_pos_deg_min = -np.array([43,40,43,40,43,40,43,40])
        self.joint_pos_deg_max = np.array([43,40,43,40,43,40,43,40])

        # Open xml
        self.xml = open(xml_path, 'rt', encoding='UTF8')
        mujoco_env.MujocoEnv.__init__(
            self,
            model_path      = self.xml_path,
            frame_skip      = self.frame_skip,
            mujoco_bindings = 'mujoco_py'
        )
        utils.EzPickle.__init__(self)
        
        # Observation and action dimension
        self.odim = self.observation_space.shape[0]
        self.adim = self.action_space.shape[0]

        # Timing
        self.HZ = int(1/self.dt)
        
        # Reset
        self.reset()
        
        # Viewer setup
        self.render_mode = render_mode
        self.render_w    = render_w
        self.render_h    = render_h
        self.render_res  = render_res
        if render_mode is not None:
            self.viewer_custom_setup(
                render_mode = render_mode,
                render_w    = render_w,
                render_h    = render_h,
                render_res  = render_res
            )
        
    def step(self,a):
        """
            Step forward
        """
        # Run sim
        self.do_simulation(a, self.frame_skip)
        # Accumulate
        self.a    = a
        self.o    = self._get_obs()
        self.r    = 0.0
        self.d    = False
        self.info = dict()
        return self.o,self.r,self.d,self.info
    
    def _get_obs(self):
        """
            Get observation
        """
        o = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext,-1,1).flat
        ])
        return o
    
    def reset_model(self):
        """
            Reset
        """
        o = np.zeros(self.odim)
        return o
    
    def get_joint_pos_deg(self):
        """
            Get joint position in [Deg]
        """
        q = self.sim.data.qpos.flat
        return np.asarray(
            [q[9],q[10],q[13],q[14],q[17],q[18],q[21],q[22]]
            )*180.0/np.pi

    def get_heading(self):
        """
            Get heading of a robot in degree
        """
        q = self.data.get_body_xquat('torso')
        _, _, z_deg = quaternion_to_euler_angle(q[0], q[1], q[2], q[3])
        return z_deg

    def get_p_torso(self):
        """
            Get torso position
        """
        p_torso = self.get_body_com("torso")
        return p_torso
    
    def get_time(self):
        """
            Get time in [Sec]
        """
        return self.sim.data.time

    def scaleup_traj(self,traj,DO_SQUASH=False,squash_margin=10):
        """
            Scale-up given joint trajectories using 'joint_pos_deg_min' and 'joint_pos_deg_max'
            It is assumed that the values in 'traj' are between -1 and +1 (but not necessarily).
            One may need to 'squash' the trajectories. 
        """
        joint_pos_center = 0.5*(self.joint_pos_deg_max+self.joint_pos_deg_min)
        joint_pos_range  = (self.joint_pos_deg_max-self.joint_pos_deg_min)
        traj_scaled      = joint_pos_center + 0.5*joint_pos_range*traj
        if DO_SQUASH:
            traj_scaled = soft_squash_multidim(
                x=traj_scaled,x_min=self.joint_pos_deg_min,x_max=self.joint_pos_deg_max,
                margin=squash_margin)
        return traj_scaled
    
    def viewer_custom_setup(
        self,
        render_mode = 'rgb_array',
        render_w    = 1500,
        render_h    = 1000,
        render_res  = 200
    ):
        """
            View setup
        """
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        self.viewer.cam.distance  = 0.7 # distance to plane (1.5)
        self.viewer.cam.elevation = -20 # elevation angle (-30)
        self.viewer.cam.lookat[0] = 0.0 # x-axis (let this follow the robot)
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 0.0
        
    def render_center(self):
        """
            Render with torso-centered
        """
        for d_idx in range(3): # follow the robot torso
            self.viewer.cam.lookat[d_idx] = self.get_body_com("torso")[d_idx]
        frame = self.render(
            mode   = self.render_mode,
            width  = self.render_w,
            height = self.render_h)
        return frame
        
# Gaussian random path 
def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in

def soft_squash_multidim(
    x      = np.random.randn(100,5),
    x_min  = -np.ones(5),
    x_max  = np.ones(5),
    margin = 0.1):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash

class GaussianRandomPathClass(object):
    def __init__(self,
                 name     = 'GRP',
                 kernel   = kernel_levse,
                 hyp_mean = {'g':1.0,'l':1.0,'w':1e-6},
                 hyp_var  = {'g':1.0,'l':1.0,'w':1e-6}
                 ):
        # super(GaussianRandomPathClass,self).__init__()
        self.name = name
        # Set kernel
        self.kernel   = kernel     # kernel function
        self.hyp_mean = hyp_mean
        self.hyp_var  = hyp_var
        # Default set data
        self.set_data()
        
    def set_data(self,
                 t_anchor     = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
                 x_anchor     = np.zeros((10,2)),
                 l_anchor     = np.ones((10,1)),
                 t_test       = np.linspace(start=0.0,stop=1.0,num=100).reshape((-1,1)),
                 l_test       = np.ones((100,1)),
                 hyp_mean     = None,
                 hyp_var      = None,
                 w_chol       = 1e-10,     # noise for Cholesky transform
                 APPLY_EPSRU  = False,     # epsilon run-up
                 t_eps        = 0.0001,
                 l_eps        = 1.0,       # leverage for epsilon run-up
                 x_diff_start = None,
                 x_diff_end   = None,
                 SKIP_GP_VAR  = False      # skip GP variance computation
                 ):
        """
            Set anchor data to GRP class
        """
        self.t_anchor    = t_anchor.astype(float)    # [N x 1]
        self.x_anchor    = x_anchor.astype(float)    # [N x D]
        self.t_test      = t_test.astype(float)      # [N_test x 1]
        self.n_anchor    = self.x_anchor.shape[0]
        self.d_anchor    = self.x_anchor.shape[1]
        self.l_anchor    = l_anchor.astype(float)    # [N x 1]
        self.n_test      = self.t_test.shape[0]
        self.l_test      = l_test.astype(float)      # [N_test x 1]
        if hyp_mean is not None: self.hyp_mean = hyp_mean
        if hyp_var is not None: self.hyp_var = hyp_var
            
        # Handle epsilon greedy algorithm
        self.APPLY_EPSRU = APPLY_EPSRU
        self.t_eps       = t_eps
        if self.APPLY_EPSRU:
            # Append runup points
            if (x_diff_start is not None) and (x_diff_end is not None):
                self.x_diff_start = x_diff_start
                self.x_diff_end   = x_diff_end
            else:
                self.x_diff_start = (self.x_anchor[-1,:]-self.x_anchor[0,:]).reshape((1,-1))
                self.x_diff_end = (self.x_anchor[-1,:]-self.x_anchor[0,:]).reshape((1,-1))
            self.t_dur  = (self.t_anchor[-1]-self.t_anchor[0]).squeeze()
            # Append 'x_anchor'
            x_anchor    = self.x_anchor
            x_anchor    = np.insert(x_anchor,1,
                                    x_anchor[0,:]+self.t_eps/self.t_dur*self.x_diff_start,axis=0)
            x_anchor    = np.insert(x_anchor,-1,
                                    x_anchor[-1,:]-self.t_eps/self.t_dur*self.x_diff_end,axis=0)
            n_anchor    = self.x_anchor.shape[0]
            # Append 'x_anchor'
            t_anchor    = self.t_anchor
            t_anchor    = np.insert(t_anchor,1,t_anchor[0,:]+self.t_eps,axis=0)
            t_anchor    = np.insert(t_anchor,-1,t_anchor[-1,:]-self.t_eps,axis=0)
            # Append 'l_anchor'
            l_anchor    = self.l_anchor
            l_eps       = 0.0
            l_anchor    = np.insert(l_anchor,1,l_eps,axis=0)
            l_anchor    = np.insert(l_anchor,-1,l_eps,axis=0)
            # Overwrite 'x_anchor', 't_anchor', 'l_anchor', and 'n_anchor'
            self.x_anchor = x_anchor
            self.t_anchor = t_anchor
            self.n_anchor = self.x_anchor.shape[0]
            if self.kernel.__name__ == 'kernel_levse': # leveraged SE kernel
                self.l_anchor = l_anchor
                
        # GP mean-related
        if self.kernel.__name__ == 'kernel_levse': # leveraged SE kernel
            l_anchor_mean = np.ones((self.n_anchor,1)) # leverage does not affect GP mean
            self.k_test_anchor_mean   = self.kernel(self.t_test,self.t_anchor,
                                                    self.l_test,l_anchor_mean,
                                                    self.hyp_mean)
            self.K_anchor_anchor_mean = self.kernel(self.t_anchor,self.t_anchor,
                                                    l_anchor_mean,l_anchor_mean,
                                                    self.hyp_mean) \
                                        + self.hyp_mean['w']*np.eye(self.n_anchor)
        elif self.kernel.__name__ == 'kernel_se': # SE kernel
            self.k_test_anchor_mean   = self.kernel(self.t_test,self.t_anchor,
                                                    self.hyp_mean)
            self.K_anchor_anchor_mean = self.kernel(self.t_anchor,self.t_anchor,
                                                    self.hyp_mean) \
                                        + self.hyp_mean['w']*np.eye(self.n_anchor)
        else:
            raise TypeError("[GaussianRandomPathClass] Unsupported kernel:[%s]"%
                            (self.kernel.__name__))
        self.x_anchor_mean        = self.x_anchor.mean(axis=0)
        self.gamma_test           = np.linalg.solve(self.K_anchor_anchor_mean,
                                                    self.x_anchor-self.x_anchor_mean)
        self.mean_test            = np.matmul(self.k_test_anchor_mean,self.gamma_test) \
                                    + self.x_anchor_mean
        
        # GP variance-related
        self.SKIP_GP_VAR = SKIP_GP_VAR
        if self.SKIP_GP_VAR: return # skip in case of computing the mean only 
        if self.kernel.__name__ == 'kernel_levse': # leveraged SE kernel
            self.k_test_test_var     = self.kernel(self.t_test,self.t_test,
                                                   self.l_test,self.l_test,
                                                   self.hyp_var)
            self.k_test_anchor_var   = self.kernel(self.t_test,self.t_anchor,
                                                   self.l_test,self.l_anchor,
                                                   self.hyp_var)
            self.K_anchor_anchor_var = self.kernel(self.t_anchor,self.t_anchor,
                                                   self.l_anchor,self.l_anchor,
                                                   self.hyp_var) \
                                        + self.hyp_mean['w']*np.eye(self.n_anchor)
        elif self.kernel.__name__ == 'kernel_se': # SE kernel
            self.k_test_test_var     = self.kernel(self.t_test,self.t_test,
                                                   self.hyp_var)
            self.k_test_anchor_var   = self.kernel(self.t_test,self.t_anchor,
                                                   self.hyp_var)
            self.K_anchor_anchor_var = self.kernel(self.t_anchor,self.t_anchor,
                                                   self.hyp_var) \
                                        + self.hyp_mean['w']*np.eye(self.n_anchor)
        else:
            raise TypeError("[GaussianRandomPathClass] Unsupported kernel:[%s]"%
                            (self.kernel.__name__))
        self.var_test            = self.k_test_test_var - np.matmul(self.k_test_anchor_var,
            np.linalg.solve(self.K_anchor_anchor_var,self.k_test_anchor_var.T))
        self.var_diag_test       = np.diag(self.var_test).reshape((-1,1))
        self.std_diag_test       = np.sqrt(self.var_diag_test)
        self.w_chol              = w_chol
        self.var_chol_test       = np.linalg.cholesky(self.var_test \
                                                      + self.w_chol*np.eye(self.n_test))
            
    def sample(self,
               n_sample  = 10,
               rand_type = 'Gaussian'):
        """
            Sample from GRP
        """
        samples = []
        for s_idx in range(n_sample):
            if rand_type == 'Gaussian':
                R = np.random.randn(self.n_test,self.d_anchor)
            elif rand_type == 'Uniform':
                rand_gain = 2 # -gain ~ +gain
                R = rand_gain*(2*np.random.rand(self.n_test,self.d_anchor)-1)
            else:
                raise TypeError("[GaussianRandomPathClass] Unsupported rand_type:[%s]"%(rand_type))
            sample = self.mean_test+np.matmul(self.var_chol_test,R)
            samples.append(sample)
        return samples,self.t_test

    def sample_one_traj(self,rand_type='Gaussian',ORG_PERTURB=False,perturb_gain=1.0):
        """
            Sample a single trajectory
        """
        if rand_type == 'Gaussian':
            R = np.random.randn(self.n_test,self.d_anchor)
        elif rand_type == 'Uniform':
            rand_gain = 2 # -gain ~ +gain
            R = rand_gain*(2*np.random.rand(self.n_test,self.d_anchor)-1)
        else:
            raise TypeError("[GaussianRandomPathClass] Unsupported rand_type:[%s]"%(rand_type))
        sample_traj = self.mean_test+np.matmul(self.var_chol_test,R) # [L x dim]
        if ORG_PERTURB:
            L = sample_traj.shape[0]
            D = sample_traj.shape[1]
            pvec = 2.0*perturb_gain*np.random.rand(1,D)-perturb_gain
            sample_traj = sample_traj + np.tile(pvec,(L,1))

        return sample_traj,self.t_test
    
    def plot(self,
             n_sample   = 10,
             ss_x_min   = None,
             ss_x_max   = None,
             ss_margin  = None,
             figsize    = (6,3),
             subplot_rc = None,
             lw_sample  = 1/4,
             ylim       = None,
             title_str  = None,
             tfs        = 10,
             rand_type  = 'Uniform',
             ):
        """
            Plot GRP
        """
        sampled_trajs,t_test = self.sample(n_sample=n_sample,rand_type=rand_type)
        colors = [plt.cm.Set1(i) for i in range(self.d_anchor)]
        if subplot_rc is not None:
            plt.figure(figsize=figsize)
        for d_idx in range(self.d_anchor):
            color = colors[d_idx]
            if subplot_rc is None:
                plt.figure(figsize=figsize)
            else:
                plt.subplot(subplot_rc[0],subplot_rc[1],d_idx+1)
            # Plot sampled trajectories
            for s_idx in range(len(sampled_trajs)):
                sampled_traj = sampled_trajs[s_idx]
                plt.plot(self.t_test,sampled_traj[:,d_idx],'-',color='k',lw=lw_sample)
            # Plot squashed trajectories
            if (ss_x_min is not None) and (ss_x_max is not None) and (ss_margin is not None):
                for s_idx in range(len(sampled_trajs)):
                    sampled_traj = sampled_trajs[s_idx]
                    squased_traj = soft_squash_multidim(
                        x      = sampled_traj,
                        x_min  = ss_x_min,
                        x_max  = ss_x_max,
                        margin = ss_margin
                        )
                    plt.plot(self.t_test,squased_traj[:,d_idx],'--',color='k',lw=lw_sample*2)
            # Plot mean
            plt.plot(self.t_test,self.mean_test[:,d_idx],'-',color=color,lw=3)
            # Plot anchors
            plt.plot(self.t_anchor,self.x_anchor[:,d_idx],'o',mfc='none',ms=10,mec='k')
            # Plot 2-standard deviation (95%)
            plt.fill_between(self.t_test.squeeze(),
                             self.mean_test[:,d_idx]-2*self.std_diag_test.squeeze(),
                             self.mean_test[:,d_idx]+2*self.std_diag_test.squeeze(),
                             facecolor=color,interpolate=True,alpha=0.2)
            if ylim is not None:
                plt.ylim(ylim)
            if title_str is None:
                plt.title("Dim-[%d]"%(d_idx),fontsize=tfs)
            else:
                plt.title("%s"%(title_str),fontsize=tfs)
            if subplot_rc is None: plt.show()
        if subplot_rc is not None: plt.show()
    
    def set_prior(
        self,
        q_init  = np.zeros(8),
        dur_sec = 2.0,
        HZ      = 50,
        hyp     = {'g':1.5,'l':1/2,'w':1e-8},
        eps_sec = 0.01):
        """
            Set GRP prior
        """
        # Eps-runup
        t_anchor = np.array([[0.0,eps_sec,dur_sec-eps_sec,dur_sec]]).T
        x_anchor = np.tile(q_init,reps=(4,1))
        l_anchor = np.array([[1,1,1,1]]).T
        n_test   = int(dur_sec*HZ)
        t_test   = np.linspace(start=0.0,stop=dur_sec,num=n_test).reshape((-1,1))
        l_test   = np.ones((n_test,1))
        self.set_data(
            t_anchor=t_anchor,x_anchor=x_anchor,l_anchor=l_anchor,
            t_test=t_test,l_test=l_test,hyp_mean=hyp,hyp_var=hyp,w_chol=1e-10)

    def interpolate(
        self,t_anchor,x_anchor,t_test,hyp={'g':1,'l':1/5,'w':1e-8},APPLY_EPSRU=True,t_eps=0.05,
        joint_pos_deg_min=None,joint_pos_deg_max=None,margin=1.0):
        """
            Interpolation
        """
        n_anchor,n_test = t_anchor.shape[0],t_test.shape[0]
        l_anchor,l_test = np.ones(shape=(n_anchor,1)),np.ones(shape=(n_test,1))
        self.set_data(t_anchor=t_anchor,x_anchor=x_anchor,l_anchor=l_anchor,t_test=t_test,l_test=l_test,
                       hyp_mean=hyp,hyp_var=hyp,
                       APPLY_EPSRU=APPLY_EPSRU,t_eps=t_eps,SKIP_GP_VAR=True)
        traj_test = self.mean_test
        if (joint_pos_deg_min is not None) and (joint_pos_deg_max is not None):
            traj_test = soft_squash_multidim(
                x=traj_test,x_min=joint_pos_deg_min,x_max=joint_pos_deg_max,margin=margin)
        return traj_test





def set_grp_prior_for_snapbot(GRP,env,max_sec=2.0,hyp={'g':1.5,'l':1/4,'w':1e-8},DO_PLOT=False):
    HZ       = int(1/env.dt)
    t_anchor = np.array([[0.0,0.01,max_sec-0.01,max_sec]]).T
    x_anchor = np.zeros(shape=(4,env.adim))
    l_anchor = np.array([[1,1,1,1]]).T
    t_test   = np.linspace(start=0.0,stop=max_sec,num=int(max_sec*HZ)).reshape((-1,1))
    l_test   = np.ones((int(max_sec*HZ),1))
    GRP.set_data(t_anchor=t_anchor,x_anchor=x_anchor,l_anchor=l_anchor,
                 t_test=t_test,l_test=l_test,hyp_mean=hyp,hyp_var=hyp,w_chol=1e-10)
    if DO_PLOT:
        GRP.plot(n_sample=20,figsize=(15,5),subplot_rc=(2,4),
                 lw_sample=1/2,tfs=10,rand_type='Uniform')

# PID Controller            
class PID_ControllerClass(object):
    def __init__(self,
                 name      = 'PID',
                 k_p       = 0.01,
                 k_i       = 0.0,
                 k_d       = 0.001,
                 dt        = 0.01,
                 dim       = 1,
                 dt_min    = 1e-6,
                 out_min   = -np.inf,
                 out_max   = np.inf,
                 ANTIWU    = True,   # anti-windup
                 out_alpha = 0.0    # output EMA (0: no EMA)
                ):
        """
            Initialize PID Controller
        """
        self.name      = name
        self.k_p       = k_p
        self.k_i       = k_i
        self.k_d       = k_d
        self.dt        = dt
        self.dim       = dim
        self.dt_min    = dt_min
        self.out_min   = out_min
        self.out_max   = out_max
        self.ANTIWU    = ANTIWU
        self.out_alpha = out_alpha
        # Buffers
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = 0.0
        self.t_prev   = 0.0
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def reset(self,t_curr=0.0):
        """
            Reset PID Controller
        """
        self.cnt      = 0
        self.x_trgt   = np.zeros(shape=self.dim)
        self.x_curr   = np.zeros(shape=self.dim)
        self.out_val  = np.zeros(shape=self.dim)
        self.out_val_prev = np.zeros(shape=self.dim)
        self.t_curr   = t_curr
        self.t_prev   = t_curr
        self.err_curr = np.zeros(shape=self.dim)
        self.err_intg = np.zeros(shape=self.dim)
        self.err_prev = np.zeros(shape=self.dim)
        self.p_term   = np.zeros(shape=self.dim)
        self.d_term   = np.zeros(shape=self.dim)
        self.err_out  = np.zeros(shape=self.dim)
        
    def update(
        self,
        t_curr  = None,
        x_trgt  = None,
        x_curr  = None,
        VERBOSE = False
        ):
        """
            Update PID controller
            u(t) = K_p e(t) + K_i int e(t) {dt} + K_d {de}/{dt}
        """
        self.cnt = self.cnt + 1
        if t_curr is not None:
            self.t_curr  = t_curr
        if x_trgt is not None:
            self.x_trgt  = x_trgt
        if x_curr is not None:
            self.x_curr  = x_curr
            # PID controller updates here
            self.dt       = max(self.t_curr - self.t_prev,self.dt_min)
            self.err_curr = self.x_trgt - self.x_curr     
            self.err_intg = self.err_intg + (self.err_curr*self.dt)
            self.err_diff = self.err_curr - self.err_prev
            
            if self.ANTIWU: # anti-windup
                self.err_out = self.err_curr * self.out_val
                self.err_intg[self.err_out<0.0] = 0.0
            
            if self.dt > self.dt_min:
                self.p_term   = self.k_p * self.err_curr
                self.i_term   = self.k_i * self.err_intg
                self.d_term   = self.k_d * self.err_diff / self.dt
                self.out_val  = np.clip(
                    a     = self.p_term + self.i_term + self.d_term,
                    a_min = self.out_min,
                    a_max = self.out_max)
                # Smooth the output control value using EMA
                self.out_val = self.out_alpha*self.out_val_prev + \
                    (1.0-self.out_alpha)*self.out_val
                self.out_val_prev = self.out_val
                
                if VERBOSE:
                    print ("[%d] t_curr:[%.2f] dt:[%.2f]"%
                           (self.cnt,self.t_curr,self.dt))
                    print (" x_trgt:   %s"%(self.x_trgt))
                    print (" x_curr:   %s"%(self.x_curr))
                    print (" err_curr: %s"%(self.err_curr))
                    print (" err_intg: %s"%(self.err_intg))
                    print (" p_term:   %s"%(self.p_term))
                    print (" i_term:   %s"%(self.i_term))
                    print (" d_term:   %s"%(self.d_term))
                    print (" out_val:  %s"%(self.out_val))
                    print (" err_out:  %s"%(self.err_out))
            
            # Backup
            self.t_prev   = self.t_curr
            self.err_prev = self.err_curr
            
    def out(self):
        """
            Get control output
        """
        return self.out_val

def plot_arrow_and_text(
    p             = np.zeros(2),
    deg           = 0.0,
    arrow_len     = 0.01,
    arrow_width   = 0.005,
    head_width    = 0.015,
    head_length   = 0.012,
    arrow_color   = 'r',
    alpha         = 0.5,
    arrow_ec      = 'none',
    arrow_lw      = 1,
    text_str      = 'arrow',
    text_x_offset = 0.0,
    text_y_offset = 0.0,
    tfs           = 10,
    text_color    = 'k',
    bbox_ec       = (0.0,0.0,0.0),
    bbox_fc       = (1.0,0.9,0.8),
    bbox_alpha    = 0.5,
    text_rotation = 0.0
    ):
    """
        Plot arrow with text
    """
    x,y = p[0],p[1]
    u,v = np.cos(deg*np.pi/180.0),np.sin(deg*np.pi/180.0)
    plt.arrow(x=x,y=y,dx=arrow_len*u,dy=arrow_len*v,
              width=arrow_width,head_width=head_width,head_length=head_length,
              color=arrow_color,alpha=alpha,ec=arrow_ec,lw=arrow_lw)
    plt.text(x=x+text_x_offset,y=y+text_y_offset,
             s=text_str,fontsize=tfs,ha='center',va='center',color=text_color,
             bbox=dict(boxstyle="round",ec=bbox_ec,fc=bbox_fc,alpha=bbox_alpha),
             rotation=text_rotation)

def process_frames(
    secs          = np.linspace(0.0,1.0,100),
    frames        = ['']*100,
    animate_every = 5
):
    """
        Preprocess frames
    """
    L = len(frames)
    frames_process,secs_process, = [],[]
    for tick in range(L):
        frame,sec = frames[tick],secs[tick]
        if (tick%animate_every) == 0:
            # Add title
            h,w = frame.shape[0],frame.shape[1]
            frame_text = np.copy(frame)
            cv2.putText(
                img              = frame_text,
                text             = '[%03d/%d] %.2fsec'%(tick,L,sec),
                org              = (int(0.25*w),int(0.09*h)),
                fontFace         = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale        = 0.5,
                color            = (255,255,255),
                thickness        = 1,
                lineType         = cv2.LINE_AA,
                bottomLeftOrigin = False)
            # Append
            frames_process.append(frame_text)
            secs_process.append(sec)
    secs_process = np.array(secs_process) # to np array
    return secs_process,frames_process

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim.to_jshtml())

def display_frames_as_gif(
    frames      = [],
    interval_ms = 20,
    figsize     = (6,4)
):
    fig = plt.figure(figsize=figsize)
    patch = plt.imshow(frames[0])
    fig.tight_layout()
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(
        fig      = plt.gcf(),
        func     = animate,
        frames   = len(frames),
        interval = interval_ms)
    display(display_animation(anim))

def animate_frames(secs,frames,animate_every=10):
    """
        Animate frames
    """
    secs_process,frames_process = process_frames(secs,frames,animate_every=animate_every)
    display_frames_as_gif(frames=frames_process,interval_ms=1000*secs_process[-1]/secs_process.shape[0])

def snapbot_zero_pose(env,PID,t_max=1.0):
    """
        Stay at zero pose
    """
    env.reset()
    PID.reset()
    for tick in range(1000):
        t_curr = env.get_time()
        PID.update(
            x_trgt = np.zeros(env.adim),
            t_curr = t_curr,
            x_curr = env.get_joint_pos_deg())
        env.step(PID.out())
        if t_curr > t_max: break
    return env,PID    

def torch2np(x_torch):
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np
def np2torch(x_np,device='cpu'):
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch

class DeepLatentPolicyGradientClass(nn.Module):
    def __init__(
        self,
        name     = 'DLPG',              
        x_dim    = 784,              # input dimension
        c_dim    = 10,               # condition dimension
        z_dim    = 16,               # latent dimension
        h_dims   = [64,32],          # hidden dimensions of encoder (and decoder)
        actv_enc = nn.ReLU(),        # encoder activation
        actv_dec = nn.ReLU(),        # decoder activation
        actv_q   = nn.Softplus(),    # q activation
        actv_out = None,             # output activation
        var_max  = None,             # maximum variance
        device   = 'cpu'
        ):
        """
            Initialize
        """
        super(DeepLatentPolicyGradientClass,self).__init__()
        self.name = name
        self.x_dim    = x_dim
        self.c_dim    = c_dim
        self.z_dim    = z_dim
        self.h_dims   = h_dims
        self.actv_enc = actv_enc
        self.actv_dec = actv_dec
        self.actv_q   = actv_q
        self.actv_out = actv_out
        self.var_max  = var_max
        self.device   = device
        # Initialize layers
        self.init_layers()
        self.init_params()
        # Test
        self.test()
        
    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        
        # Encoder part
        h_dim_prev = self.x_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims):
            self.layers['enc_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            self.layers['enc_%02d_actv'%(h_idx)] = \
                self.actv_enc
            h_dim_prev = h_dim
        self.layers['z_mu_lin']  = nn.Linear(h_dim_prev,self.z_dim,bias=True)
        self.layers['z_var_lin'] = nn.Linear(h_dim_prev,self.z_dim,bias=True)
        
        # Decoder part
        h_dim_prev = self.z_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims[::-1]):
            self.layers['dec_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            self.layers['dec_%02d_actv'%(h_idx)] = \
                self.actv_dec
            h_dim_prev = h_dim
        self.layers['out_lin'] = nn.Linear(h_dim_prev,self.x_dim,bias=True)
        
        # Append parameters
        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.cvae_parameters = nn.ParameterDict(self.param_dict)
        
    def xc_to_z_mu(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_mu
        """
        if c is not None:
            net = torch.cat((x,c),dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            net = self.layers['enc_%02d_actv'%(h_idx)](net)
        z_mu = self.layers['z_mu_lin'](net)
        return z_mu
    
    def xc_to_z_var(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_var
        """
        if c is not None:
            net = torch.cat((x,c),dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            net = self.layers['enc_%02d_actv'%(h_idx)](net)
        net = self.layers['z_var_lin'](net)
        if self.var_max is None:
            net = torch.exp(net)
        else:
            net = self.var_max*torch.sigmoid(net)
        z_var = net
        return z_var
    
    def zc_to_x_recon(
        self,
        z = torch.randn(2,16),
        c = torch.randn(2,10)
        ):
        """
            z and c to x_recon
        """
        if c is not None:
            net = torch.cat((z,c),dim=1)
        else:
            net = z
        for h_idx,_ in enumerate(self.h_dims[::-1]):
            net = self.layers['dec_%02d_lin'%(h_idx)](net)
            net = self.layers['dec_%02d_actv'%(h_idx)](net)
        net = self.layers['out_lin'](net)
        if self.actv_out is not None:
            net = self.actv_out(net)
        x_recon = net
        return x_recon
    
    def xc_to_z_sample(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_sample
        """
        z_mu,z_var = self.xc_to_z_mu(x=x,c=c),self.xc_to_z_var(x=x,c=c)
        eps_sample = torch.randn(
            size=z_mu.shape,dtype=torch.float32).to(self.device)
        z_sample   = z_mu + torch.sqrt(z_var+1e-10)*eps_sample
        return z_sample
    
    def xc_to_x_recon(
        self,
        x             = torch.randn(2,784),
        c             = torch.randn(2,10), 
        STOCHASTICITY = True
        ):
        """
            x and c to x_recon
        """
        if STOCHASTICITY:
            z_sample = self.xc_to_z_sample(x=x,c=c)
        else:
            z_sample = self.xc_to_z_mu(x=x,c=c)
        x_recon = self.zc_to_x_recon(z=z_sample,c=c)
        return x_recon
    
    def sample_x(
        self,
        c             = torch.randn(5,10),
        n_sample      = 5,
        SKIP_Z_SAMPLE = False
        ):
        """
            Sample x
        """
        z_sample = torch.randn(
            size=(n_sample,self.z_dim),dtype=torch.float32).to(self.device)
        if SKIP_Z_SAMPLE:
            return self.zc_to_x_recon(z=z_sample,c=c)
        else:
            return self.zc_to_x_recon(z=z_sample,c=c),z_sample
    
    def init_params(self,seed=0):
        """
            Initialize parameters
        """
        # Fix random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Init
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def test(
        self,
        batch_size = 4
        ):
        """
            Unit tests
        """
        x_test   = torch.randn(batch_size,self.x_dim)
        if self.c_dim > 0:
            c_test = torch.randn(batch_size,self.c_dim)
        else:
            c_test = None
        z_test   = torch.randn(batch_size,self.z_dim)
        z_mu     = self.xc_to_z_mu(x=x_test,c=c_test)
        z_var    = self.xc_to_z_var(x=x_test,c=c_test)
        x_recon  = self.zc_to_x_recon(z=z_test,c=c_test)
        z_sample = self.xc_to_z_sample(x=x_test,c=c_test)
        x_recon  = self.xc_to_x_recon(x=x_test,c=c_test)
    
    def loss_recon(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        q               = torch.ones(2),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0,
        STOCHASTICITY   = True
        ):
        """
            Recon loss
        """
        x_recon = self.xc_to_x_recon(x=x,c=c,STOCHASTICITY=STOCHASTICITY)
        if (LOSS_TYPE == 'L1') or (LOSS_TYPE == 'MAE'):
            errs = torch.mean(torch.abs(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L2') or (LOSS_TYPE == 'MSE'):
            errs = torch.mean(torch.square(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L1+L2') or (LOSS_TYPE == 'EN'):
            errs = torch.mean(
                0.5*(torch.abs(x-x_recon)+torch.square(x-x_recon)),axis=1)
        else:
            raise Exception("VAE:[%s] Unknown loss_type:[%s]"%
                            (self.name,LOSS_TYPE))
        # Weight errors by q
        if self.actv_q is not None: q = self.actv_q(q)
        errs = errs*q # [N]
        return recon_loss_gain*torch.mean(errs)
    
    def loss_kl(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10),
        q = torch.randn(2)
        ):
        """
            KLD loss
        """
        z_mu     = self.xc_to_z_mu(x=x,c=c)
        z_var    = self.xc_to_z_var(x=x,c=c)
        z_logvar = torch.log(z_var)
        errs     = 0.5*torch.sum(z_var + z_mu**2 - 1.0 - z_logvar,axis=1)
        # Weight errors by q
        if self.actv_q is not None: q = self.actv_q(q)
        errs     = errs*q # [N]
        return torch.mean(errs)
        
    def loss_total(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        q               = torch.ones(2),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0,
        STOCHASTICITY   = True,
        beta            = 1.0
        ):
        """
            Total loss
        """
        loss_recon_out = self.loss_recon(
            x               = x,
            c               = c,
            q               = q,
            LOSS_TYPE       = LOSS_TYPE,
            recon_loss_gain = recon_loss_gain,
            STOCHASTICITY   = STOCHASTICITY
        )
        loss_kl_out    = beta*self.loss_kl(
            x = x,
            c = c,
            q = q
        )
        loss_total_out = loss_recon_out + loss_kl_out
        info           = {'loss_recon_out' : loss_recon_out,
                          'loss_kl_out'    : loss_kl_out,
                          'loss_total_out' : loss_total_out,
                          'beta'           : beta}
        return loss_total_out,info
    
    def debug_plot_img(
        self,
        x_train_np     = np.zeros((60000,784)),  # to plot encoded latent space 
        y_train_np     = np.zeros((60000)),      # to plot encoded latent space 
        c_train_np     = np.zeros((60000,10)),   # to plot encoded latent space
        x_test_np      = np.zeros((10000,784)),
        c_test_np      = np.zeros((10000,10)),
        c_vecs         = np.eye(10,10),
        n_sample       = 10,
        img_shape      = (28,28),
        img_cmap       = 'gray',
        figsize_image  = (10,3.25),
        figsize_latent = (10,3.25),
        DPP_GEN        = False,
        dpp_hyp        = {'g':1.0,'l':0.1}
        ):
        """
            Debug plot
        """
        n_train            = x_train_np.shape[0]
        z_prior_np         = np.random.randn(n_train,self.z_dim)
        x_train_torch      = np2torch(x_train_np)
        c_train_torch      = np2torch(c_train_np)
        z_mu_train_np      = torch2np(self.xc_to_z_mu(x_train_torch,c_train_torch))
        z_sample_train_out = torch2np(
            self.xc_to_z_sample(x_train_torch,c_train_torch))
        
        # Reconstruct
        x_test_torch       = np2torch(x_test_np)
        c_test_torch       = np2torch(c_test_np)
        n_test             = x_test_np.shape[0]
        rand_idxs          = np.random.permutation(n_test)[:n_sample]
        if self.c_dim > 0:
            x_recon = self.xc_to_x_recon(
                x = x_test_torch[rand_idxs,:],
                c = c_test_torch[rand_idxs,:]).detach().cpu().numpy()
        else:
            x_recon = self.xc_to_x_recon(
                x = x_test_torch[rand_idxs,:],
                c = None).detach().cpu().numpy()
        
        # Plot images to reconstruct
        fig = plt.figure(figsize=figsize_image)
        for s_idx in range(n_sample):
            plt.subplot(1,n_sample,s_idx+1)
            plt.imshow(x_test_np[rand_idxs[s_idx],:].reshape(img_shape),
                       vmin=0,vmax=1,cmap=img_cmap)
            plt.axis('off')
        fig.suptitle("Images to Reconstruct",fontsize=15);plt.show()
        
        # Plot reconstructed images
        fig = plt.figure(figsize=figsize_image)
        for s_idx in range(n_sample):
            plt.subplot(1,n_sample,s_idx+1)
            plt.imshow(x_recon[s_idx,:].reshape(img_shape),
                       vmin=0,vmax=1,cmap=img_cmap)
            plt.axis('off')
        fig.suptitle("Reconstructed Images",fontsize=15);plt.show()
        
        # Plot conditioned generated images
        if DPP_GEN:
            n_sample_total = 100
            z_sample_total = np.random.randn(n_sample_total,self.z_dim)
            z_sample,_ = ikdpp(
                xs_total = z_sample_total,
                qs_total = None,
                n_select = n_sample,
                n_trunc  = n_sample_total,
                hyp      = dpp_hyp)
        else:
            z_sample = np.random.randn(n_sample,self.z_dim)
        z_sample_torch = np2torch(z_sample)
        c_vecs_torch   = np2torch(c_vecs)
        
        # Plot (conditioned) generated images
        if self.c_dim > 0:
            for r_idx in range(c_vecs.shape[0]):
                c_torch  = c_vecs_torch[r_idx,:].reshape((1,-1))
                c_np     = c_vecs[r_idx,:]
                fig      = plt.figure(figsize=figsize_image)
                for s_idx in range(n_sample):
                    z_torch = z_sample_torch[s_idx,:].reshape((1,-1))
                    x_recon = self.zc_to_x_recon(z=z_torch,c=c_torch)
                    x_reocn_np = torch2np(x_recon)
                    plt.subplot(1,n_sample,s_idx+1)
                    plt.imshow(x_reocn_np.reshape(img_shape),vmin=0,vmax=1,cmap=img_cmap)
                    plt.axis('off')
                fig.suptitle("Conditioned Generated Images c:%s"%
                             (c_np),fontsize=15);plt.show()
        else:
            fig = plt.figure(figsize=figsize_image)
            for s_idx in range(n_sample):
                z_torch = z_sample_torch[s_idx,:].reshape((1,-1))
                x_recon = self.zc_to_x_recon(z=z_torch,c=None)
                x_reocn_np = torch2np(x_recon)
                plt.subplot(1,n_sample,s_idx+1)
                plt.imshow(x_reocn_np.reshape(img_shape),vmin=0,vmax=1,cmap=img_cmap)
                plt.axis('off')
            fig.suptitle("Generated Images",fontsize=15);plt.show()
            
        # Plot latent space of training inputs
        fig = plt.figure(figsize=figsize_latent)
        # Plot samples from the prior
        plt.subplot(1,3,1) # z prior
        plt.scatter(z_prior_np[:,0],z_prior_np[:,1],marker='.',s=0.5,c='k',alpha=0.5)
        plt.title('z prior',fontsize=13)
        plt.xlim(-3,+3); plt.ylim(-3,+3)
        plt.gca().set_aspect('equal', adjustable='box')
        # Plot encoded mean
        plt.subplot(1,3,2) # z mu
        plt.scatter(
            x      = z_mu_train_np[:,0],
            y      = z_mu_train_np[:,1],
            c      = y_train_np,
            cmap   = 'rainbow',
            marker = '.',
            s      = 0.5,
            alpha  = 0.5)
        plt.title('z mu',fontsize=13)
        plt.xlim(-3,+3); plt.ylim(-3,+3)
        plt.gca().set_aspect('equal', adjustable='box')
        # Plot samples
        plt.subplot(1,3,3) # z sample
        sc = plt.scatter(
            x      = z_sample_train_out[:,0],
            y      = z_sample_train_out[:,1],
            c      = y_train_np,
            cmap   = 'rainbow',
            marker = '.',
            s      = 0.5,
            alpha  = 0.5)
        plt.plot(z_sample[:,0],z_sample[:,1],'o',mec='k',mfc='none',ms=10)
        colors = [plt.cm.rainbow(a) for a in np.linspace(0.0,1.0,10)]
        for c_idx in range(10):
            plt.plot(10,10,'o',mfc=colors[c_idx],mec=colors[c_idx],ms=6,
                     label='%d'%(c_idx)) # outside the axis, only for legend
        plt.legend(fontsize=8,loc='upper right')
        plt.title('z sample',fontsize=13)
        plt.xlim(-3,+3); plt.ylim(-3,+3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show() # plot latent spaces
        
def plot_snapbot_traj_topdown_view(p_torsos,z_degs,secs,figsize=(8,8),title_str='Snapbot Trajectory (Top View)'):
    # Plot snapbot xy trajectory
    plt.figure(figsize=figsize)
    plt.plot(p_torsos[:,0],p_torsos[:,1],'-',lw=2,color='k')
    n_arrow = 10
    colors = [plt.cm.Spectral(i) for i in np.linspace(1,0,n_arrow)]
    max_tick = p_torsos.shape[0]
    scale = 10.0*max(p_torsos.max(axis=0)-p_torsos.min(axis=0))
    for t_idx,tick in enumerate(
        np.linspace(start=0,stop=max_tick-1,num=n_arrow).astype(np.int32)):
        p,deg    = p_torsos[tick,0:2],z_degs[tick]
        text_str = '[%d] %.1fs'%(tick,secs[tick])
        plot_arrow_and_text(p=p,deg=deg,text_str=text_str,tfs=8,
                            arrow_color=colors[t_idx],arrow_ec='none',
                            alpha=0.2,text_x_offset=0.0,text_y_offset=0.0,
                            arrow_len=0.01*scale,arrow_width=0.005*scale,
                            head_width=0.015*scale,head_length=0.012*scale,
                            text_rotation=deg,bbox_fc='w',bbox_ec='k')
    # Initial and final pose
    plot_arrow_and_text(p=p_torsos[0,0:2],deg=z_degs[0],text_str='Start',tfs=15,
                        arrow_color=colors[0],arrow_ec='k',arrow_lw=2,
                        alpha=0.9,text_x_offset=0.0,text_y_offset=0.0075*scale,
                        arrow_len=0.01*scale,arrow_width=0.005*scale,
                        head_width=0.015*scale,head_length=0.012*scale,
                        text_rotation=z_degs[0],bbox_fc='w',bbox_ec='k',text_color='b')
    plot_arrow_and_text(p=p_torsos[-1,0:2],deg=z_degs[-1],text_str='Final',tfs=15,
                        arrow_color=colors[-1],arrow_ec='k',arrow_lw=2,
                        alpha=0.9,text_x_offset=0.0,text_y_offset=0.0075*scale,
                        arrow_len=0.01*scale,arrow_width=0.005*scale,
                        head_width=0.015*scale,head_length=0.012*scale,
                        text_rotation=z_degs[-1],bbox_fc='w',bbox_ec='k',text_color='b')
    plt.axis('equal'); plt.grid('on')
    plt.title(title_str,fontsize=15)
    plt.show()

def plot_snapbot_joint_traj_and_topdown_traj(
    traj_secs,traj_joints,t_anchor,x_anchor,xydegs,secs,
    figsize=(16,8),title_str='Snapbot Trajectory (Top View)',tfs=15
    ):
    """
        Plot Snapbot joint trajectories and topdown-view trajectory
    """
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1); # joint trajectory
    n_joint = traj_joints.shape[1]
    colors = [plt.cm.rainbow(a) for a in np.linspace(0.0,1.0,n_joint)]
    for t_idx in range(n_joint):
        plt.plot(traj_secs,traj_joints[:,t_idx],'-',color=colors[t_idx],label='Joint %d'%(t_idx))
    for t_idx in range(n_joint):
        plt.plot(t_anchor,x_anchor[:,t_idx],'o',color=colors[t_idx])
    plt.xlabel('Time (sec)',fontsize=13)
    plt.ylabel('Position (deg)',fontsize=13)
    plt.legend(fontsize=10,loc='upper left')
    plt.title('Joint Trajectories',fontsize=tfs)

    plt.subplot(1,2,2); # top-down view
    p_torsos = xydegs[:,:2]
    z_degs   = xydegs[:,2]
    plt.plot(p_torsos[:,0],p_torsos[:,1],'-',lw=2,color='k')
    n_arrow = 10
    colors = [plt.cm.Spectral(i) for i in np.linspace(1,0,n_arrow)]
    max_tick = p_torsos.shape[0]
    scale = 10.0*max(p_torsos.max(axis=0)-p_torsos.min(axis=0))
    for t_idx,tick in enumerate(
        np.linspace(start=0,stop=max_tick-1,num=n_arrow).astype(np.int32)):
        p,deg    = p_torsos[tick,0:2],z_degs[tick]
        text_str = '[%d] %.1fs'%(tick,secs[tick])
        plot_arrow_and_text(p=p,deg=deg,text_str=text_str,tfs=8,
                            arrow_color=colors[t_idx],arrow_ec='none',
                            alpha=0.2,text_x_offset=0.0,text_y_offset=0.0,
                            arrow_len=0.01*scale,arrow_width=0.005*scale,
                            head_width=0.015*scale,head_length=0.012*scale,
                            text_rotation=deg,bbox_fc='w',bbox_ec='k')
    # Initial and final pose
    plot_arrow_and_text(p=p_torsos[0,0:2],deg=z_degs[0],text_str='Start',tfs=15,
                        arrow_color=colors[0],arrow_ec='k',arrow_lw=2,
                        alpha=0.9,text_x_offset=0.0,text_y_offset=0.0075*scale,
                        arrow_len=0.01*scale,arrow_width=0.005*scale,
                        head_width=0.015*scale,head_length=0.012*scale,
                        text_rotation=z_degs[0],bbox_fc='w',bbox_ec='k',text_color='b')
    plot_arrow_and_text(p=p_torsos[-1,0:2],deg=z_degs[-1],text_str='Final',tfs=15,
                        arrow_color=colors[-1],arrow_ec='k',arrow_lw=2,
                        alpha=0.9,text_x_offset=0.0,text_y_offset=0.0075*scale,
                        arrow_len=0.01*scale,arrow_width=0.005*scale,
                        head_width=0.015*scale,head_length=0.012*scale,
                        text_rotation=z_degs[-1],bbox_fc='w',bbox_ec='k',text_color='b')
    plt.axis('equal'); plt.grid('on')
    plt.title('Top-down Torso Trajectory',fontsize=tfs)
    plt.suptitle(title_str,fontsize=tfs)
    plt.show()

def get_anchors_from_traj(t_test,traj,n_anchor=20):
    """
    Get equidist anchors from a trajectory
    """
    n_test = len(t_test)
    idxs = np.round(np.linspace(start=0,stop=n_test-1,num=n_anchor)).astype(np.int16)
    t_anchor,x_anchor = t_test[idxs],traj[idxs]
    return t_anchor,x_anchor

class NormalizerClass(object):
    def __init__(self,
                 name    = 'NZR',
                 x       = np.random.rand(100,4),
                 eps     = 1e-6,
                 axis    = 0,     # mean and std axis (0 or None)
                 CHECK   = True,
                 VERBOSE = False):
        super(NormalizerClass,self).__init__()
        self.name    = name
        self.x       = x
        self.eps     = eps
        self.axis    = axis
        self.CHECK   = CHECK
        self.VERBOSE = VERBOSE
        # Set data
        self.set_data(x=self.x,eps=self.eps)
        
    def set_data(self,x=np.random.rand(100,4),eps=1e-6):
        """
            Set data
        """
        self.mean = np.mean(x,axis=self.axis)
        self.std  = np.std(x,axis=self.axis)
        if np.min(self.std) < 1e-4:
            self.eps = 1.0 # numerical stability
        # Check
        if self.CHECK:
            x_org        = self.x
            x_nzd        = self.get_nzd_data(x_org=x_org)
            x_org2       = self.get_org_data(x_nzd=x_nzd)
            x_err        = x_org - x_org2
            max_abs_err  = np.max(np.abs(x_err)) # maximum absolute error
            mean_abs_err = np.mean(np.abs(x_err),axis=None) # mean absolute error
            if self.VERBOSE:
                print ("[NormalizerClass][%s] max_err:[%.3e] min_err:[%.3e]"%
                    (self.name,max_abs_err,mean_abs_err))
            
    def get_nzd_data(self,x_org):
        x_nzd = (x_org-self.mean)/(self.std + self.eps)
        return x_nzd
    
    def get_org_data(self,x_nzd):
        x_org = x_nzd*(self.std + self.eps) + self.mean
        return x_org

def whitening(x=np.random.rand(100,2)):
    """
        Whitening
    """
    if len(x.shape) == 1:
        x_mean  = np.mean(x,axis=None)
        x_std   = np.std(x,axis=None)
    else:
        x_mean  = np.mean(x,axis=0)
        x_std   = np.std(x,axis=0)
    return (x-x_mean)/x_std

def whitening_torch(x=np.random.rand(100,2)):
    """
        Whitening
    """
    if len(x.shape) == 1:
        x_mean  = torch.mean(x)
        x_std   = torch.std(x)
    else:
        x_mean  = torch.mean(x,axis=0)
        x_std   = torch.std(x,axis=0)
    return (x-x_mean)/x_std    

class TicTocClass():
    def __init__(self,name='tictoc'):
        """
            Initialize tic-toc class
        """
        self.name = name
        self.t_start = time.time()

    def toc(self):
        self.t_elapsed = time.time() - self.t_start
        return self.t_elapsed

    
