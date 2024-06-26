import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import plotly.express as px
from tqdm import tqdm
import pickle as pkl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import os

from discriminator import Discriminator

class Generator():

    def __init__(self, discriminator):
        self.fear_data = np.load('Training_data/fear_data.npy')
        self.no_fear_data = np.load('Training_data/no_fear_data.npy')
        
        with open('Training_data/fear_vel.pkl', 'rb') as f:
            self.fear_traj = pkl.load(f)

        with open('Training_data/bas_vel.pkl', 'rb') as f:
            self.no_fear_traj = pkl.load(f)

        # trajectory blending parameters
        self.lookback = 20
        self.maxlen = 200
        self.buffer = deque(maxlen = self.maxlen)
        # self.discriminator = tf.keras.models.load_model(discriminator)
        self.discriminator = Discriminator()
        self.discriminator.define_model()
        self.discriminator.model.load_weights(discriminator)
        self.discriminator = self.discriminator.model
        self.input_size = self.discriminator.input_shape[2]
        self.reference_profile = self.no_fear_data[10, :, :]

        # table generation parameters
        self.num_points = 50
        self.alphas = np.linspace(0, 1, self.num_points, dtype=np.float32)
        self.prob_upper_bound = 0.9
        self.prob_lower_bound = 0.1

        self.blending_indexes = []

    def GPU_generate_tables(self):
        # Set the device to GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Define the device
        with tf.device("/gpu:0"):
            fear = tf.stack(self.fear_data)
            no_fear = tf.stack(self.no_fear_data)

            alphas = tf.Variable(self.alphas, dtype=tf.float32)

            alphas = tf.reshape(alphas, [-1, 1, 1])
            
            result = alphas * fear + (1 - alphas) * no_fear
        result = result.numpy()
        
    def generate_tables(self):
        self.blending_indexes = np.zeros((self.fear_data.shape[0], self.no_fear_data.shape[0]))
        print('generating a {0} x {1} belending coefficient table'.format(self.blending_indexes.shape[0], self.blending_indexes.shape[1]))
        for z in tqdm(range(self.fear_data.shape[0])):
            self.reference_profile = self.fear_data[z, :, :]
            blending_data = np.zeros((1, self.no_fear_data.shape[1], self.no_fear_data.shape[2]))
            fear_index = np.zeros((self.num_points, self.no_fear_data.shape[0]))
            
            # evaluate fear index for every velocity profile
            for j in range(self.no_fear_data.shape[0]):
                for i in range(self.num_points):
                    blending_data[0, :, :] = self.alphas[i] * self.no_fear_data[j, :, :] + (1 - self.alphas[i]) * self.reference_profile
                    fear_index[i, j] = self.discriminator(blending_data[:1]).numpy()
                index = np.where(fear_index[:, j] >= self.prob_upper_bound) 
                if len(index[0]) == 0:
                    index = np.where(fear_index[:, j] >= fear_index[0, j] * self.prob_upper_bound)[0]
                index = np.max(index)

                self.blending_indexes[z, j] = index

            # print('|--------------PROFILE {0} ACQUIRED--------------|'.format(z+1))

            # show results for debugging purposes
            
            '''
            plt.figure('fear exploration')
            plt.imshow(fear_index, cmap='hot', interpolation='nearest', aspect='auto', origin='lower', extent=[0, self.num_points, 0, self.fear_data.shape[0]])
            plt.colorbar(label='fear index')
            plt.ylabel('alpha')
            plt.xlabel('velocity profile (no fear)')

            plt.savefig('Figures/fear_exploration.png')

            plt.figure('alpha indexes')
            plt.plot(temp_indexes)
            plt.ylabel('alpha indexes')
            plt.xlabel('velocity profile index')

            plt.savefig('Figures/alpha_indexes.png')

            plt.figure('blending coefficients')
            plt.plot(self.alphas[temp_indexes])
            plt.ylabel('blending coefficients')
            plt.xlabel('velocity profile index')

            plt.savefig('Figures/blending_coefficients.png')

            plt.show()
            '''

        np.save('Trained_models/blending_indexes_table.npy', self.blending_indexes)

    def check_similarity(self, index):
        distances_x = np.zeros(len(self.no_fear_traj))
        distances_y = np.zeros(len(self.fear_traj))

        for i in range(len(self.no_fear_traj)):
            distances_x[i] = np.mean(np.linalg.norm(self.no_fear_traj[i][:, index-len(self.buffer):index].transpose() - self.buffer, axis=1))

        for i in range(len(self.fear_traj)):
            distances_y[i] = np.mean(np.linalg.norm(self.fear_traj[i][:, index-len(self.buffer):index].transpose() - self.buffer, axis=1))

        return np.argmin(distances_x), np.argmin(distances_y)

    def blend(self, L0_point, alpha, t):
        # set a maximum speed variation
        buffer = np.reshape(self.buffer, [3, len(self.buffer)])
        delta_v_b = np.array([np.max(np.abs(buffer[0, len(buffer) - min(len(buffer),self.lookback):len(self.buffer)-2] - buffer[0, len(buffer) + 1 - min(len(buffer),self.lookback):len(self.buffer)-1])), 
                              np.max(np.abs(buffer[1, len(buffer) - min(len(buffer),self.lookback):len(self.buffer)-2] - buffer[1, len(buffer) + 1 - min(len(buffer),self.lookback):len(self.buffer)-1])),
                              np.max(np.abs(buffer[2, len(buffer) - min(len(buffer),self.lookback):len(self.buffer)-2] - buffer[2, len(buffer) + 1 - min(len(buffer),self.lookback):len(self.buffer)-1]))])
        
        delta_v = np.max(delta_v_b)

        L2_point = alpha * L0_point + (1 - alpha) * self.reference_profile[:, t]

        return L2_point

    def add_data(self, data):
        self.buffer.append(data)
    
    def reset_buffer(self):
        self.buffer.clear()

    def set_blending_indexes(self, indexes):
        self.blending_indexes = indexes

    def set_reference_profile(self, reference_profile):
        self.reference_profile = reference_profile

    def show_blending_table(self):
        matrix = np.zeros_like(self.blending_indexes)

        for i in range(matrix.shape[0]):
            matrix[i, :] = self.alphas[[ int(x) for x in self.blending_indexes[i, :]]]

        row_means = np.mean(matrix, axis = 1)
        col_means = np.mean(matrix, axis = 0)

        row_indices = np.argsort(row_means)
        col_indices = np.argsort(col_means)

        matrix = matrix[:, col_indices]
        matrix = matrix[row_indices, :]

        fig = px.imshow(matrix, color_continuous_scale='Viridis', origin='lower')

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='no fear',
            yaxis_title='fear'
        )

        fig.update_xaxes(showline=True, linecolor='black', gridcolor='black')
        fig.update_yaxes(showline=True, linecolor='black', gridcolor='black')

        fig.update_xaxes(zeroline=True, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinecolor='black')
        
        fig.write_image("Figures/pdf/blending_indexes.pdf")

        fig.show()

        fig = go.Figure(go.Scatter( x=np.array(range(0, matrix.shape[0])), y=np.mean(matrix.transpose(), axis=0), mode="markers"))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='no fear'
        )

        fig.update_xaxes(showline=True, linecolor='black', gridcolor='black')
        fig.update_yaxes(showline=True, linecolor='black', gridcolor='black')

        fig.update_xaxes(zeroline=True, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinecolor='black')
        
        fig.write_image("Figures/pdf/crushed_blending_indexes.pdf")

        fig.show()

    def normalize_length(self, data, size = 3, val = 0, num = 200, flag = True):

        for i in range(len(data)):
            temp = np.zeros([size, num])
            temp[:, 0:max(data[i].shape)] = data[i]
            temp[:, max(data[i].shape):] = val
            data[i] = temp

        if flag:
            for i in range(len(self.fear_traj)):
                temp = np.zeros([3, num])
                temp[:, 0:self.fear_traj[i].shape[1]] = self.fear_traj[i]
                self.fear_traj[i] = temp

            for i in range(len(self.no_fear_traj)):
                temp = np.zeros([3, num])
                temp[:, 0:self.no_fear_traj[i].shape[1]] = self.no_fear_traj[i]
                self.no_fear_traj[i] = temp

        return data
    
    def evaluate_information(self, data):
        time = data.shape[1]
        eval = np.zeros((1, data.shape[0], self.input_size))
        f_x = interp1d(np.linspace(0, time - 1, time), data[0 , 0:time], kind = 'cubic')
        f_y = interp1d(np.linspace(0, time - 1, time), data[1 , 0:time], kind = 'cubic')
        f_z = interp1d(np.linspace(0, time - 1, time), data[2 , 0:time], kind = 'cubic')
        eval[0, 0, :] = f_x(np.linspace(0, time - 1, self.input_size))
        eval[0, 1, :] = f_y(np.linspace(0, time - 1, self.input_size))
        eval[0, 2, :] = f_z(np.linspace(0, time - 1, self.input_size))
        return self.discriminator(eval[:1]).numpy()
    
    def main_visualizer(self, data, predictions = 0, distances = 0, goal = 0, points = 100):
        # matrix = []
        time = data[0].shape[0]
        row_means = [data[0].mean()]
        for i in range(1, len(data)):  
            time = max(data[i].shape[0], time)
            row_means.append(data[i].mean())

        matrix = self.normalize_length(data, 1, np.nan, time, False)
        matrix = np.array([matrix[i].reshape(-1) for i in range(len(matrix))])

        row_indices = np.argsort(row_means)

        matrix = matrix[row_indices, :]

        fig = px.imshow(matrix, color_continuous_scale='Viridis', zmin=0, zmax=1, origin='lower')

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        fig.write_image("Figures/online_coeff.pdf")
        fig.show()

        predictions = np.array(predictions).flatten().reshape([len(predictions), 1])

        fig = px.imshow(np.array(predictions[row_indices[::-1], :]), color_continuous_scale=[(0, 'white'), (1, 'green')])

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale = False,
        )

        fig.update_yaxes(visible=False, gridcolor='black')
        fig.update_xaxes(visible=False, gridcolor='black')
        
        fig.update_xaxes(zeroline=True, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinecolor='black')

        fig.write_image("Figures/pdf/online_preditions.pdf")
        fig.show()

        distances = np.array(distances).flatten().reshape([len(distances), 1])

        fig = px.imshow(distances[row_indices[::-1], :] <= goal, color_continuous_scale=[(0, 'white'), (1, 'purple')])

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale = False,
        )

        fig.update_yaxes(visible=False, gridcolor='grey')
        fig.update_xaxes(visible=False, gridcolor='grey')
        
        fig.update_xaxes(zeroline=True, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinecolor='black')

        fig.write_image("Figures/pdf/online_distances.pdf")
        fig.show()

        print('distance succes rate: {}%'.format(100 * np.mean(distances[row_indices[::-1], :] <= goal)))

    def show_amp_profiles(self, fear_traj, no_fear_traj, amp_traj):
        mean_fear = np.mean(fear_traj, axis=0)
        std_fear = np.std(fear_traj, axis=0)

        mean_no_fear = np.mean(no_fear_traj, axis=0)
        std_no_fear = np.std(no_fear_traj, axis=0)

        mean_amp_fear = np.mean(amp_traj, axis=0)
        std_amp_fear = np.std(amp_traj, axis=0)

        shade = 0.0001

        fig = make_subplots(rows=3, cols=1)

        for ax_ind in range(0, 3):
            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), y = mean_fear[ax_ind, :] + std_fear[ax_ind, :],
                                    mode='lines', line=dict(color='green',width =shade), legendgroup='fear', showlegend=False), row=ax_ind + 1, col=1)
            
            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), 
                                            y=mean_fear[ax_ind, :], marker=dict(color='green', ), fill='tonexty', legendgroup='fear', showlegend=True), row=ax_ind + 1, col=1)

            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), y = mean_fear[ax_ind, :] - std_fear[ax_ind, :], fill='tonexty',
                                    mode='lines', line=dict(color='green', width =shade), legendgroup='fear', showlegend=False), row=ax_ind + 1, col=1)
             
            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), y = mean_no_fear[ax_ind, :] + std_no_fear[ax_ind, :],
                                    mode='lines', line=dict(color='red',width =shade), legendgroup='no fear', showlegend=False), row=ax_ind + 1, col=1)

            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), 
                                            y=mean_no_fear[ax_ind, :], marker=dict(color='red', ), fill='tonexty', legendgroup='no fear', showlegend=True), row=ax_ind + 1, col=1)

            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), y = mean_no_fear[ax_ind, :] - std_no_fear[ax_ind, :], fill='tonexty',
                                    mode='lines', line=dict(color='red',width =shade), legendgroup='no fear', showlegend=False), row=ax_ind + 1, col=1)
            
            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), y = mean_amp_fear[ax_ind, :] + std_amp_fear[ax_ind, :],
                                    mode='lines', line=dict(color='blue',width =shade), legendgroup='amplified', showlegend=False), row=ax_ind + 1, col=1)
            
            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), 
                                            y=mean_amp_fear[ax_ind, :], marker=dict(color='blue', ), fill='tonexty',  legendgroup='amplified', showlegend=True), row=ax_ind + 1, col=1)

            fig.append_trace(go.Scatter(x=np.linspace(0, self.maxlen, self.maxlen), y = mean_amp_fear[ax_ind, :] - std_amp_fear[ax_ind, :], fill='tonexty',
                                    mode='lines', line=dict(color='blue',width =shade),  legendgroup='amplified', showlegend=False), row=ax_ind + 1, col=1)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        fig.update_xaxes(showline=True, linecolor='black')
        fig.update_yaxes(showline=True, linecolor='black')

        fig['layout']['xaxis3']['title']='samples'
        fig['layout']['yaxis']['title']='x - velocity'
        fig['layout']['yaxis2']['title']='y - velocity'
        fig['layout']['yaxis3']['title']='z - velocity'

        fig.write_image("Figures/pdf/amp_velocities.pdf")
        fig.show()

def main():
    discriminator = 'Trained_models/discriminator_model.keras'
    gen = Generator(discriminator)
    # gen.set_blending_indexes(np.load('Trained_models/blending_indexes_table.npy'))
    gen.generate_tables()
    gen.show_blending_table()


if __name__ == '__main__':
    main()
