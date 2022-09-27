import wandb, os
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from class_snapbot import Snapbot4EnvClass
from class_vqvae import VectorQuantizedVariationalAutoEncoder
from utils import *
from class_pid import PIDControllerClass
from class_grp import GaussianRandomPathClass, scaleup_traj, get_anchors_from_traj
from class_dlpg import DeepLatentPolicyGradientClass

class Eval():
    def __init__(self,
            name = "SnapbotTrajectoryUpdateClass",
            env      = None,
            z_dim    = 32,
            c_dim    = 3,
            h_dims   = [128, 128],
            embedding_num  = 200,
            embedding_dim  = 32,
            tau_scale = 1.0,
            kld_scale = 5e-4,
            n_anchor = 20,
            max_repeat    = 5,
            device_idx = 0,
            VERBOSE    = True
            ):
        # Init params
        self.env        = env
        self.name       = name
        self.z_dim      = z_dim
        self.c_dim      = c_dim
        self.n_anchor   = n_anchor
        self.max_repeat = max_repeat
        # try: 
        #     self.device  = torch.device('mps')
        # except:
        self.device  = torch.device('cuda:{}'.format(device_idx) if torch.cuda.is_available() else 'cpu')
        self.VERBOSE     = VERBOSE
        # Set grp & pid & qscaler
        self.DLPG  = VectorQuantizedVariationalAutoEncoder(name='GQVAE', x_dim=env.adim*n_anchor, c_dim=c_dim, z_dim=z_dim, h_dims=h_dims, \
                                                            embedding_num=embedding_num, embedding_dim=embedding_dim, tau_scale=tau_scale, kld_scale=kld_scale, \
                                                            actv_enc=nn.ReLU(), actv_dec=nn.ReLU(), actv_q=nn.Softplus(), actv_out=None, device=self.device)
        self.DLPG.to(self.device)
        # Set explaination
        if self.VERBOSE:
            print("{} START with DEVICE: {}".format(self.name, self.device))

if __name__ == "__main__":
    env = Snapbot4EnvClass(render_mode=None)
    EvalClass = Eval(
                    name = "EvalClass",
                    env      = env,
                    z_dim    = 32,
                    c_dim    = 3,
                    h_dims   = [128, 128],
                    embedding_num  = 100,
                    embedding_dim  = 32,
                    tau_scale = 1.0,
                    kld_scale = 2e-4,
                    n_anchor = 20,
                    max_repeat    = 5,
                    device_idx = 0,
                    VERBOSE    = True
                    )
    
    print('load')
    epoch = 90
    EvalClass.DLPG.load_state_dict(torch.load("dlpg/2003/weights/dlpg_model_weights_{}.pth".format(epoch), map_location=torch.device('cpu')))
    # Get Evaluate Trajectory from File
    x = torch.load('dlpg/2003/batch/x_{}.pth'.format(epoch))
    c = torch.load('dlpg/2003/batch/c_{}.pth'.format(epoch))
    q = torch.load('dlpg/2003/batch/scaled_q_{}.pth'.format(epoch))
    
    # recon
    recon = EvalClass.DLPG.xc_to_x_recon(x=torch.tensor(x, dtype=torch.float32),c=torch.tensor(c, dtype=torch.float32))
    print(recon)
    # plotting
    t = np.linspace(0, 2, 20)
    for i in range(250):
        test_recon = recon[i].reshape(20, EvalClass.env.adim)
        test_x = x[i].reshape(20, EvalClass.env.adim)
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t, test_recon.detach().numpy())
        plt.subplot(2,1,2)
        plt.plot(t, test_x)

        plt.show()



    