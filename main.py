import numpy as np
from generator import Generator
from discriminator import Discriminator
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.graph_objs import Layout
import os 
from fine_tuner import Fine_Tuner
from tqdm import tqdm
from collections import deque

import tensorflow as tf
import pickle as pkl

# monitoring
import time
import psutil

pio.renderers.default = "browser"
pio.kaleido.scope.mathjax = None

def compute_moving_average(data, size, length):
    aver_reward = np.zeros(length - size)
    aver = deque(data[0:size].tolist(), maxlen=size)
    epsilons = np.ones(length)

    for j in range(1, length): epsilons[j] = max(0.001, 0.995 * epsilons[j-1])

    for i in range(size, length):
        aver_reward[i - size] = np.mean(aver)
        aver.append(data[i])

    return aver_reward, epsilons
    
if __name__ == '__main__':

    discriminator = 'Trained_models/discriminator_model.keras'
    generator = Generator(discriminator)

    indexes = np.load('Trained_models/blending_indexes_table.npy')
    generator.set_blending_indexes(indexes)

    with open('Training_data/val_bas_vel.pkl', 'rb') as f:
        no_fear_traj = pkl.load(f)
    no_fear_traj = generator.normalize_length(no_fear_traj)
    no_fear_traj_lengths = np.load('Training_data/val_bas_vel_len.npy')
    no_fear_vel_ID = np.load('Training_data/val_bas_vel_ID.npy')
    no_fear_vel_trial_ID = np.load('Training_data/val_bas_vel_trial_ID.npy')
    
    N_no_fear = len(no_fear_traj)

    # FEAR PROFILE
    with open('Training_data/fear_vel.pkl', 'rb') as f:
        fear_traj = pkl.load(f)
    fear_traj = generator.normalize_length(fear_traj)
    fear_traj_lengths = np.load('Training_data/fear_vel_len.npy')
    fear_vel_ID = np.load('Training_data/fear_vel_ID.npy')
    fear_vel_trial_ID = np.load('Training_data/fear_vel_trial_ID.npy')
    N_fear = len(no_fear_traj)

    amp_traj = []
    amp_traj_lengths = np.zeros_like(no_fear_traj_lengths)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    agent = Fine_Tuner(optimizer, batch_size=128, ki=6)

    agent.brain_target.load_weights('Trained_models/policy.keras')

    N_steps = max([np.max(fear_traj_lengths), np.max(no_fear_traj_lengths)])

    print('Mib used {0}'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    [disc_reward, epsilons] = compute_moving_average(np.load('Trained_models/discounted_reward.npy'), 100, 1300)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(0, len(disc_reward))), y=disc_reward, showlegend=False))
    fig.add_trace(go.Scatter(x=np.array(range(0, len(disc_reward))), y=agent.sigma * np.ones(len(disc_reward)), line=dict(color='purple', ), showlegend=False))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis = dict(range=[0, 1000]),
    )

    fig.update_xaxes(showline=True, linecolor='black')
    fig.update_yaxes(showline=True, linecolor='black')
    fig.write_image("Figures/aver_reward.pdf")
    fig.show()

    success_matrix = np.zeros((no_fear_traj_lengths.shape[0], fear_traj_lengths.shape[0]))

    errors = np.zeros((no_fear_traj_lengths.shape[0] * fear_traj_lengths.shape[0], generator.input_size))

    generator.show_blending_table()

    predictions = []
    comp_times = []
    distances = []
    alpha_list = []

    for exec_ind in tqdm(range(len(no_fear_traj))):

        start_time = time.time()

        generator.reset_buffer()

        L0_traj = no_fear_traj[exec_ind]
        L0_final_time = no_fear_traj_lengths[exec_ind] 

        prediction = generator.evaluate_information(L0_traj[:, 0:L0_final_time])

        L2_traj = np.zeros_like(L0_traj)
        a = 1
        alpha = [a]
        pendency_blend = 5
        count = pendency_blend

        # INTEGRATE TO GET POSITIONS

        Ts = 0.01
        L2_pos = np.zeros_like(L0_traj)
        L0_pos = np.zeros_like(L0_traj)
        ref_pos = np.zeros_like(L0_traj)

        x0 = np.array([0, 0, 0])
        L2_pos[:, 0] = x0
        L0_pos[:, 0] = x0
        ref_pos[:, 0] = x0

        x_f = L0_pos[:, -1]

        a_count = 0

        alpha_up = 10

        state = np.zeros(agent.state_size)
        terminated = 0  # DEBUG
        time_in = 0
        exit_times = 0
        total_reward = 0
        disc_tot_reward = 0
        done = False
        exp_trace = []
        exp = np.array([0, 0, 0])

        for t in range(alpha_up):
            L2_traj[:, t + 1] = L0_traj[:, t + 1]
            L2_pos[:, t + 1] = L2_traj[:, t + 1] * Ts + L2_pos[:, t]
            L0_pos[:, t + 1] = L0_traj[:, t + 1] * Ts + L0_pos[:, t]
            alpha.append(a)
            generator.add_data(L2_traj[:, t])
        
        x_index = generator.check_similarity(alpha_up)[1]
        generator.set_reference_profile(fear_traj[x_index])
        agent.ks = np.array(max(fear_traj_lengths[x_index], no_fear_traj_lengths[exec_ind]))
        agent.calculate_correction()
        horizon = int(max(fear_traj_lengths[x_index], no_fear_traj_lengths[exec_ind]))

        for t in range(alpha_up):
            ref_pos[:, t + 1] = generator.reference_profile[:, t + 1] * Ts + ref_pos[:, t]

        state = [L0_pos[:, t+1], L2_pos[:, t+1], L0_traj[:, t+1], L2_traj[:, t+1], agent.ks]

        t = alpha_up

        for t in range(alpha_up, horizon):
            y_index = generator.check_similarity(t)[0]
            if t % alpha_up == 0:
                a = generator.alphas[int(generator.blending_indexes[x_index, y_index])]
                action = agent.deploy_action(state)
                exp = agent.actions[action]

            L2_traj[:, t + 1] = generator.blend(L0_traj[:, t], alpha[-1], t) + exp * agent.ki * (L0_pos[:, t] - L2_pos[:, t])    # decay shoud come from RL fine tuner
            L2_pos[:, t + 1] = L2_traj[:, t + 1] * Ts + L2_pos[:, t]
            L0_pos[:, t + 1] = L0_traj[:, t + 1] * Ts + L0_pos[:, t]
            ref_pos[:, t + 1] = generator.reference_profile[:, t + 1] * Ts + ref_pos[:, t]

            exp_trace.append(exp)

            next_state = [L0_pos[:, t+1], L2_pos[:, t+1], L0_traj[:, t+1], L2_traj[:, t+1], agent.ks]

            reward, in_flag = agent.reward_wrapper(0, state, next_state, np.sum(exp), terminated)

            if in_flag == 1:
                # L2 enterd the goal region
                time_in += 1

            total_reward += reward
            disc_tot_reward += agent.gamma ** (t - alpha_up) * reward

            state = next_state

            alpha.append(a)
            generator.add_data(L2_traj[:, t])
            t+=1


        amp_traj.append(L2_traj[:, 0:horizon])
        amp_traj_lengths[exec_ind] = horizon
            
        alpha_list.append(np.array(alpha))

        comp_times.append(time.time() - start_time)

        prediction = generator.evaluate_information(L2_traj[:, 0:fear_traj_lengths[x_index]])
        predictions.append(prediction > 0.5)
        success_matrix[exec_ind, x_index] = prediction

        distances.append(np.linalg.norm(L0_pos[:, horizon -1] - L2_pos[:, horizon -1]))

        f = interp1d(np.linspace(0, horizon - 1, horizon), np.linalg.norm(L0_pos[:, 0:horizon] - L2_pos[:, 0:horizon], axis=0), kind = 'cubic')
        errors[(x_index + 1) * exec_ind, :] = f(np.linspace(0, horizon - 1, generator.input_size))

        # plot velocity profiles [plotly]
        fig = make_subplots(rows=4, cols=1)
        
        legends = (False, False, False)
        legend_names = ('L0', 'R', 'L2')
        colors = ('red', 'green', 'blue')
        traj = (L0_traj, generator.reference_profile, L2_traj)
        exp_trace = np.array(exp_trace)

        for i in range(1, 4):
            for j in range(3):
                fig.append_trace(go.Scatter(
                x=np.array(range(0, horizon)),
                y=traj[j][i-1, 0:horizon],
                line=dict(color=colors[j], ), 
                name=legend_names[j],
                legendgroup=legend_names[j],
                showlegend=legends[i-1],
                ), row=i, col=1)

        fig.append_trace(go.Scatter(
            x=np.array(range(0, horizon)),
            y=alpha[0:horizon],
            line=dict(color='black', ), 
            name=legend_names[j],
            legendgroup=legend_names[j],
            showlegend=legends[i-1],
            ), row=4, col=1)

        fig.update_layout(title="SUBJECT {}-{}, BLENDED WITH SUBJECT {}-{} PREDICITON {}".format(no_fear_vel_ID[exec_ind], no_fear_vel_trial_ID[exec_ind], fear_vel_ID[x_index], fear_vel_trial_ID[x_index], prediction[0][0]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis4 = dict(range=[0, 1])
        )

        fig.update_xaxes(showline=True, linecolor='black', gridcolor='black')
        fig.update_yaxes(showline=True, linecolor='black', gridcolor='black')

        fig.update_xaxes(zeroline=True, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinecolor='black')

        if fear_vel_ID[x_index] == no_fear_vel_ID[exec_ind]: fig.write_image("Figures/same_subject/pdf/fear_Velocity_{}_{}.pdf".format(x_index, exec_ind))
        else: fig.write_image("Figures/pdf/fear_Velocity_{}_{}.pdf".format(x_index, exec_ind))

        # plot 3D trajectory
        ms_anim = 1   
        delta_axis = 20

        #start plotting
        fig = go.Figure(
            data=[go.Scatter3d(x=L0_pos[0, 0:horizon], y=L0_pos[1, 0:horizon], z=L0_pos[2, 0:horizon],
                            mode="markers",
                            name="L0",
                            legendgroup="L0",
                            marker=dict(color='purple', )),
                    go.Scatter3d(x=ref_pos[0, 0:horizon], y=ref_pos[1, 0:horizon], z=ref_pos[2, 0:horizon],
                            mode="markers",
                            name="R",
                            legendgroup="R",
                            marker=dict(color='green', )),
                    go.Scatter3d(x=L2_pos[0, 0:horizon], y=L2_pos[1, 0:horizon], z=L2_pos[2, 0:horizon],
                            mode="markers",
                            name="L2",
                            legendgroup="L2",
                            marker=dict(color='blue', )),
                    go.Scatter3d(x=L0_pos[0, 0:horizon], y=L0_pos[1, 0:horizon], z=L0_pos[2, 0:horizon],
                            mode="lines",
                            name="L0",
                            legendgroup="L0",
                            showlegend=False,
                            line=dict(color='purple', )),
                    go.Scatter3d(x=ref_pos[0, 0:horizon], y=ref_pos[1, 0:horizon], z=ref_pos[2, 0:horizon],
                            mode="lines",
                            name="R",
                            legendgroup="R",
                            showlegend=False,
                            line=dict(color='green', )),
                    go.Scatter3d(x=L2_pos[0, 0:horizon], y=L2_pos[1, 0:horizon], z=L2_pos[2, 0:horizon],
                            mode="lines",
                            name="L2",
                            legendgroup="L2",
                            showlegend=False,
                            line=dict(color='blue', ))
                ],
        #add button in layout
            layout=go.Layout(title="SUBJECT {}-{}, BLENDED WITH SUBJECT {}-{} PREDICITON {}".format(no_fear_vel_ID[exec_ind], no_fear_vel_trial_ID[exec_ind], fear_vel_ID[x_index], fear_vel_trial_ID[x_index], prediction[0][0]),
                            # plot_bgcolor='rgba(0,0,0,0)',
                            hovermode="closest",
                            updatemenus=[dict(type="buttons",
                                            buttons=[dict(label="Play",
                                                            method="animate",
                                                            args=[None, {"frame": {"duration": ms_anim, "redraw": False},}])])],
                            scene = dict(
                                xaxis=dict(range=[-delta_axis + min(np.min(L0_pos[0, 0:horizon]), np.min(ref_pos[0, 0:horizon]), np.min(L2_pos[0, 0:horizon])), 
                                    delta_axis + max(np.max(L0_pos[0, 0:horizon]), np.max(ref_pos[0, 0:horizon]), np.max(L2_pos[0, 0:horizon]))], ),
                                yaxis=dict(range=[-delta_axis + min(np.min(L0_pos[1, 0:horizon]), np.min(ref_pos[1, 0:horizon]), np.min(L2_pos[1, 0:horizon])), 
                                    delta_axis + max(np.max(L0_pos[1, 0:horizon]), np.max(ref_pos[1, 0:horizon]), np.max(L2_pos[1, 0:horizon]))], ), 
                                zaxis=dict(range=[-delta_axis + min(np.min(L0_pos[2, 0:horizon]), np.min(ref_pos[2, 0:horizon]), np.min(L2_pos[2, 0:horizon])), 
                                    delta_axis + max(np.max(L0_pos[2, 0:horizon]), np.max(ref_pos[2, 0:horizon]), np.max(L2_pos[2, 0:horizon]))], ),
                                    aspectmode="cube")
                            ),
        #pass frames
            frames=[go.Frame(
                data=[go.Scatter3d(
                    x=L0_pos[0, k:k+1],
                    y=L0_pos[1, k:k+1],
                    z=L0_pos[2, k:k+1],
                    ),
                    go.Scatter3d(
                    x=ref_pos[0, k:k+1],
                    y=ref_pos[1, k:k+1],
                    z=ref_pos[2, k:k+1],
                    ),
                    go.Scatter3d(
                    x=L2_pos[0, k:k+1],
                    y=L2_pos[1, k:k+1],
                    z=L2_pos[2, k:k+1],
                    ),
                    go.Scatter3d(
                    x=L0_pos[0, :k],
                    y=L0_pos[1, :k],
                    z=L0_pos[2, :k],
                    ),
                    go.Scatter3d(
                    x=ref_pos[0, :k],
                    y=ref_pos[1, :k],
                    z=ref_pos[2, :k],
                    ),
                    go.Scatter3d(
                    x=L2_pos[0, :k],
                    y=L2_pos[1, :k],
                    z=L2_pos[2, :k],
                    )
                ]) for k in range(horizon)]
        )

        for button in fig.layout.updatemenus[0].buttons:
            button['args'][1]['frame']['redraw'] = True
            
        fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

        if fear_vel_ID[x_index] == no_fear_vel_ID[exec_ind]: fig.write_html("Figures/same_subject/Positions/fear_Position_{}_{}.html".format(x_index, exec_ind))
        else: fig.write_html("Figures/Positions/fear_Position_{}_{}.html".format(x_index, exec_ind))
        # fig.show()

    print('success rate: {0}%'.format(100 * np.mean(predictions)))

    print('Mib used {0}'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    print('Min lag {0}, Max lag {1}'.format(min(comp_times), max(comp_times)))

    disk_usage = psutil.disk_usage('/')
    print(f"Disk Usage: {disk_usage.percent}%")

# spatial error [plotly]
norm_x = np.linspace(0, 1, num=generator.input_size) 
mean_error = np.mean(errors, axis=0)
std_error = np.std(errors, axis=0)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=norm_x,
    y=mean_error,
    line=dict(color='blue', ), 
    name='spatial error with human trajectory',
    showlegend=False,))

fig.add_trace(go.Scatter(
    x=np.concatenate([norm_x, (norm_x)[::-1]]),  # x, then x reversed
    y=np.concatenate([np.clip(mean_error - std_error, 0, None), (mean_error + std_error)[::-1]]),  # upper, then lower reversed
    fill='toself',
    fillcolor='rgba(0, 0, 255, 0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False,))

fig.add_trace(go.Scatter(x=norm_x, y=agent.theta_goal * np.ones(generator.input_size),
                         line=dict(color='purple', ),
                         showlegend=False,))

fig.update_layout(
    title='Fear profile played: {}'.format(exec_ind),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font = dict(size=30)
)

fig.update_xaxes(showline=True, linecolor='black', gridcolor='black')
fig.update_yaxes(showline=True, linecolor='black', gridcolor='black')

fig.update_xaxes(zeroline=True, zerolinecolor='black')
fig.update_yaxes(zeroline=True, zerolinecolor='black')

fig.write_image("Figures/pdf/Average_Error.pdf")
fig.show()

generator.show_amp_profiles(fear_traj, no_fear_traj, generator.normalize_length(amp_traj))

generator.main_visualizer(alpha_list, predictions, distances, agent.theta_goal)

print('fear mean duration {}'.format(np.mean(fear_traj_lengths)))
print('no fear mean duration {}'.format(np.mean(no_fear_traj_lengths)))
print('amp mean duration {}'.format(np.mean(amp_traj_lengths)))
print('maximum distance {}'.format(max(distances)))
print('mean distance {}'.format(np.mean(np.array(distances))))
