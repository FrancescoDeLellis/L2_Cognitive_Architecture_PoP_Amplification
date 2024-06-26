import numpy as np
from generator import Generator
from discriminator import Discriminator
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.graph_objs import Layout
import os 
import pickle as pkl
from tqdm import tqdm

import tensorflow as tf
from collections import deque
import random

# monitoring
import psutil

pio.renderers.default = "browser"
pio.kaleido.scope.mathjax = None

class Fine_Tuner():
    def __init__(self, optimizer, batch_size, ki):
        self.actions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        # self.actions = np.array([[0, 0, 0], [1, 1, 1]])

        self.state_size = 13
        self.action_size = self.actions.shape[0]
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.omega = 0.8

        # experience replay
        self.replay_exp = deque(maxlen=100000)

        self.gamma = 0.99  # discount factor 
        self.epsilon = 1   # exploration intialization

        # Build Policy Network
        self.brain_policy = tf.keras.models.Sequential()
        self.brain_policy.add(tf.keras.layers.Dense(128, input_dim=self.state_size, activation="relu"))
        self.brain_policy.add(tf.keras.layers.Dense(128, activation="relu"))
        self.brain_policy.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        self.brain_policy.compile(loss="mse", optimizer=self.optimizer)

        # Build Target Network
        self.brain_target = tf.keras.models.Sequential()
        self.brain_target.add(tf.keras.layers.Dense(128, input_dim=self.state_size, activation="relu"))
        self.brain_target.add(tf.keras.layers.Dense(128, activation="relu"))
        self.brain_target.add(tf.keras.layers.Dense(self.action_size, activation="linear"))
        self.brain_target.compile(loss="mse", optimizer=self.optimizer)

        self.update_brain_target()

        # set goal region to stabilize
        self.theta_goal = 20

        self.vel_goal = 0.1

        # set desired settling time (maximum number of steps required to reach goal region)
        self.ks = 100
        # set out time horizon
        self.kout = 200

        # estimation of bounds of the reward function in and outside the goal region
        self.r_max = 0
        self.r_min = -30
        self.r_G_max = 0
        self.r_G_min = -30

        # set objective value treshold (modulate to fulfill Corollary IV.7)
        self.sigma = 10000

        # calulate prize and punishment according to Theorem IV.5 (Assumption IV.3)
        self.prize = self.sigma * (1 - self.gamma) / self.gamma ** (self.ks) - self.r_max * (1 - self.gamma ** (self.ks)) / self.gamma ** (self.ks) - self.r_G_max

        self.punishment = self.sigma * self.gamma ** (-self.kout) + \
                          self.r_max / (1 - self.gamma) - \
                          (self.r_G_max + self.prize) * (self.gamma ** (-self.kout) - 1) / (1 - self.gamma)
        
        self.ki = ki

        self.ep_start = 0

    def calculate_correction(self):
        # calulate prize and punishment according to Theorem IV.5 (Assumption IV.3)
        self.prize = self.sigma * (1 - self.gamma) / self.gamma ** (self.ks) - self.r_max * (1 - self.gamma ** (self.ks)) / self.gamma ** (self.ks) - self.r_G_max

        self.punishment = self.sigma * self.gamma ** (-self.kout) + \
                                self.r_max / (1 - self.gamma) - \
                                (self.r_G_max + self.prize) * (self.gamma ** (-self.kout) - 1) / (1 - self.gamma)
        
    def memorize_exp(self, state, action, reward, next_state, done):
        self.replay_exp.append((state, action, reward, next_state, done))

    def update_brain_target(self):
        return self.brain_target.set_weights(self.brain_policy.get_weights())
    
    def load_checkpoint(self, path='./Data'):
        self.brain_policy = tf.keras.models.load_model(path + '/policy.keras')
        self.update_brain_target()

        with open(path + '/buffer.pkl', 'rb') as file:
            agent.replay_exp = pkl.load(file)

        r = np.load(path + '/reward.npy')
        d = np.load(path + '/discounted_reward.npy') 
        a = np.load(path + '/aver.npy')
        a = deque(a.tolist(), maxlen = len(a))
        a_r = np.load(path + '/aver_reward.npy')
        s = np.load(path + '/start.npy')
        self.epsilon = float(np.load(path + '/epsilon.npy'))

        return r, d, a, a_r, s

    # Choosing action according to epsilon-greedy policy
    def choose_action_DQN(self, state, flag):
        state = np.reshape(np.concatenate(state, axis=None), [1, self.state_size])
        qhat = self.brain_policy(state).numpy()
        action = np.argmax(qhat[0])

        random = np.random.random()
        if flag == 0:
            if random > self.epsilon:
                return action
            else:
                return np.random.choice(self.action_size)
        else:
            return action

    # Deploy action according to a learned policy
    def deploy_action(self, state):
        state = np.reshape(np.concatenate(state, axis=None), [1, self.state_size])
        qhat = self.brain_target(state).numpy()
        return np.argmax(qhat[0])

    # Update parameters of neural policy
    def learn(self):
        # take a mini-batch from replay experience
        cur_batch_size = min(len(self.replay_exp), self.batch_size)
        mini_batch = random.sample(self.replay_exp, cur_batch_size)

        # batch data
        sample_states = np.ndarray(shape=(cur_batch_size, self.state_size))         # replace 128 with cur_batch_size
        sample_actions = np.ndarray(shape=(cur_batch_size, 1))
        sample_rewards = np.ndarray(shape=(cur_batch_size, 1))
        sample_next_states = np.ndarray(shape=(cur_batch_size, self.state_size))
        sample_dones = np.ndarray(shape=(cur_batch_size, 1))

        temp = 0
        for exp in mini_batch:
            sample_states[temp] = exp[0]
            sample_actions[temp] = exp[1]
            sample_rewards[temp] = exp[2]
            sample_next_states[temp] = exp[3]
            sample_dones[temp] = exp[4]
            temp += 1
        
        sample_qhat_next = self.brain_target(sample_next_states).numpy()

        sample_qhat_next = sample_qhat_next * (np.ones(shape=sample_dones.shape) - sample_dones)
        sample_qhat_next = np.max(sample_qhat_next, axis=1)

        sample_qhat = self.brain_policy(sample_states).numpy()
        
        for i in range(cur_batch_size):
            a = sample_actions[i, 0]
            sample_qhat[i, int(a)] = sample_rewards[i] + self.gamma * sample_qhat_next[i]

        q_target = sample_qhat

        self.brain_policy.fit(sample_states, q_target, epochs=1, verbose=0)

    # apply reward correction according to Assumption IV.5
    def reward_wrapper(self, reward, state, state_next, action, terminated):
        flag = 0
        next_cond = np.linalg.norm(state_next[0] - state_next[1]) <= self.theta_goal and np.linalg.norm(state_next[2]) <= self.vel_goal
        cond = np.linalg.norm(state[0] - state[1]) <= self.theta_goal and np.linalg.norm(state[2]) <= self.vel_goal
        reward = - 0.01 * np.linalg.norm(state_next[0] - state_next[1]) - 10 * np.sum(self.actions[action])
        if next_cond == True:
            reward += self.prize
            flag = 1
        elif cond == True and next_cond == False:
            reward += self.punishment
            flag = 2

        return reward, flag
    
if __name__ == '__main__':

    discriminator = '/home/redolaptop/Documents/generative_optimization_trajectory/Trained_models/discriminator_model.keras'
    generator = Generator(discriminator)

    indexes = np.load('/home/redolaptop/Documents/generative_optimization_trajectory/Trained_models/blending_indexes_table.npy')
    generator.set_blending_indexes(indexes)

    with open('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/bas_vel.pkl', 'rb') as f:
        no_fear_traj = pkl.load(f)
    no_fear_traj = generator.normalize_length(no_fear_traj)
    no_fear_traj_lengths = np.load('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/bas_vel_len.npy')
    no_fear_vel_ID = np.load('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/bas_vel_ID.npy')
    no_fear_vel_trial_ID = np.load('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/bas_vel_trial_ID.npy')
    
    N_no_fear = len(no_fear_traj)

    # FEAR PROFILE
    with open('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/fear_vel.pkl', 'rb') as f:
        fear_traj = pkl.load(f)
    fear_traj = generator.normalize_length(fear_traj)
    fear_traj_lengths = np.load('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/fear_vel_len.npy')
    fear_vel_ID = np.load('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/fear_vel_ID.npy')
    fear_vel_trial_ID = np.load('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/fear_vel_trial_ID.npy')
    N_fear = len(no_fear_traj)

    comp_times = []

    # setup the Fine Tuner training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    agent = Fine_Tuner(optimizer, batch_size=1024, ki=6)

    N_steps = max([np.max(fear_traj_lengths), np.max(no_fear_traj_lengths)])

    Episodes = 1501
    start = 0

    predictions = np.zeros(Episodes)

    rewards = np.zeros(Episodes)
    discounted_rewards = np.zeros(Episodes)
    aver_reward = np.zeros(Episodes)
    aver = deque(maxlen=50)

    print('Mib used {0}'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    exp_trace = np.zeros([3, Episodes])

    [rewards[0:1001], discounted_rewards[0:1001], aver, aver_reward[0:1001], start] = agent.load_checkpoint()

    print('START AT {} EPISODES'.format(start))
    
    for e in tqdm(range(start, Episodes)):
        # fear_traj.shape[0] # DEBUG

        exec_ind = random.randint(0, no_fear_traj_lengths.size - 1)

        print('|------------STARTING SIMULATION {0}------------| LENGTH: {1}'.format(exec_ind, agent.kout))

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
        agent.ks = max(1.1 * max(no_fear_traj_lengths), fear_traj_lengths[x_index])
        agent.calculate_correction()
        horizon = int(max(fear_traj_lengths[x_index], no_fear_traj_lengths[exec_ind]))

        for t in range(alpha_up):
            ref_pos[:, t + 1] = generator.reference_profile[:, t + 1] * Ts + ref_pos[:, t]

        state = [L0_pos[:, t+1], L2_pos[:, t+1], L0_traj[:, t+1], L2_traj[:, t+1], agent.ks]

        t = alpha_up

        while not done:
            y_index = generator.check_similarity(t)[0]
            if t % alpha_up == 0:
                action = agent.choose_action_DQN(state, terminated)
                exp = agent.actions[action]
                a = generator.alphas[int(generator.blending_indexes[x_index, y_index])]
            #     exp_trace[t] = exp
            #     count = 0
            #     f_a = interp1d([t, t + pendency_blend], [alpha[-1], a], kind = 'slinear')
            # if count < pendency_blend:
            #     a = f_a(t)
            #     count += 1
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
                done = t == agent.kout - 2
            elif in_flag == 2:
                # L2 exited the goal region
                exit_times += 1
                done = True
            else:
                done = t == agent.kout - 2

            agent.memorize_exp(np.concatenate(state, axis=None), action, reward, np.concatenate(next_state, axis=None), done)
            total_reward += reward
            disc_tot_reward += agent.gamma ** (t - alpha_up) * reward

            agent.learn()

            state = next_state
            
            alpha.append(a)
            generator.add_data(L2_traj[:, t])
            t+=1

        if e % 100 == 0:
            np.save("./Data/reward.npy", rewards)
            np.save("./Data/discounted_reward.npy", discounted_rewards)
            np.save("./Data/aver.npy", aver)
            np.save("./Data/aver_reward.npy", aver_reward)
            np.save("./Data/start.npy", e)
            np.save("./Data/epsilon.npy", agent.epsilon)
            pkl.dump(agent.replay_exp, open('./Data/buffer.pkl', 'wb'))
            agent.brain_policy.save("./Data/policy.keras")

        aver.append(disc_tot_reward)    
        aver_reward[e] = np.mean(aver)

        rewards[e] = total_reward
        discounted_rewards[e] = disc_tot_reward

        # update model_target after each episode
        agent.update_brain_target()
                
        agent.epsilon = max(0.001, 0.995 * agent.epsilon)  # decaying exploration
        print("Episode {0} | reward {1} | steps {2} | time in goal {3} | exited {4} | disc reward {5} | distance from target {6} | use of integral action {7}".format(e, total_reward, t, 
                                                                                                              time_in, exit_times, discounted_rewards[e], np.linalg.norm(L0_pos[:, agent.kout -1] - L2_pos[:, agent.kout -1]), np.mean(exp_trace)))

        prediction = generator.evaluate_information(L2_traj[:, 0:fear_traj_lengths[x_index]])
        predictions[e] = (np.round(prediction, 0))
        # print('L2 fear_level in session {0}: {1}'.format(exec_ind, prediction))

    np.save("./Data/reward.npy", rewards)
    np.save("./Data/discounted_reward.npy", discounted_rewards)
    np.save("./Data/aver.npy", aver)
    np.save("./Data/aver_reward.npy", aver_reward)
    np.save("./Data/start.npy", Episodes)
    np.save("./Data/epsilon.npy", agent.epsilon)
    pkl.dump(agent.replay_exp, open('./Data/buffer.pkl', 'wb'))
    agent.brain_policy.save("./Data/policy.keras")

    print('success rate: {0}%'.format(100 * np.sum(predictions) / Episodes))
    plt.plot(aver_reward)
    plt.show()

    print('Mib used {0}'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2))

    # print('Min lag {0}, Max lag {1}'.format(min(comp_times), max(comp_times)))

    disk_usage = psutil.disk_usage('/')
    # print(f"Disk Usage: {disk_usage.percent}%")

    # SHOW DATA
    fig = make_subplots(rows=3, cols=1)
    
    legends = (False, False, False)
    legend_names = ('L0', 'R', 'L2')
    colors = ('red', 'green', 'blue')
    traj = (L0_traj, generator.reference_profile, L2_traj)

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

    fig.update_layout(
        # title='Fear profile played: {}'.format(exec_ind),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # font=dict(size=30)
    )

    fig.update_xaxes(showline=True, linecolor='black', gridcolor='black')
    fig.update_yaxes(showline=True, linecolor='black', gridcolor='black')

    fig.update_xaxes(zeroline=True, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinecolor='black')

    fig['layout']['xaxis3']['title']='samples(sampling time 1ms)'
    fig['layout']['yaxis']['title']='x - velocity'
    fig['layout']['yaxis2']['title']='y - velocity'
    fig['layout']['yaxis3']['title']='z - velocity'

    fig.show()

    ms_anim = 1    # 1 ms between each frame 
    delta_axis = 20

    # start plotting
    fig = go.Figure(
        data=[go.Scatter3d(x=L0_pos[0, 0:horizon], y=L0_pos[1, 0:horizon], z=L0_pos[2, 0:horizon],
                        mode="markers",
                        name="L0",
                        legendgroup="L0",
                        marker=dict(color='red', )),
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
                        line=dict(color='red', )),
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
    # add button in layout
        layout=go.Layout(title="Position trajectory {}, fear prediction {}".format(exec_ind, prediction[0][0]),
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
    # pass frames
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
        
    # for k in range(len(fig.frames)):
    #     fig.frames[k]['layout'].update(title_text=r'$Validation\ movie,\ phases\ (rho_g={})$'.format(rho_g[k-1]))
        
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

    # fig.write_html("Figures/Position_{}_{}.html".format(x_index, exec_ind))

    fig.show()

