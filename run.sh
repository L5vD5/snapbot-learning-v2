python main.py \
--env 3 \
--torque 2 \
--z_dim 32 \
--c_dim 3 \
--h_dims 128 128 \
--var_max -1 \
--n_anchor 20 \
--dur_sec 2 \
--max_repeat 5 \
--lbtw_base 0.8 \
--device_idx 0 \
--seed 0 \
--start_epoch 0 \
--max_epoch 300 \
--n_sim_roll 100 \
--sim_update_size 64 \
--n_sim_update 64 \
--n_sim_prev_consider 10 \
--n_sim_prev_best_q 50 \
--init_prior_prob 0.5 \
--folder 23 \
--wandb True