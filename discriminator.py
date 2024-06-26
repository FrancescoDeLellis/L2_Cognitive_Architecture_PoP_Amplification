import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from collections import deque
import pickle as pkl
import data_loader
from tqdm import tqdm

pio.renderers.default = "browser"
pio.kaleido.scope.mathjax = None

class Discriminator():

    def __init__(self):
        
        # self.fear_data = scipy.io.loadmat('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/fear_velocity_data_v2.mat')['fear_data']
        # self.no_fear_data = scipy.io.loadmat('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/no_fear_velocity_data_v2.mat')['no_fear_data']
        # self.fear_traj = scipy.io.loadmat('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/fear_velocity_traj_v3.mat')['input_data']
        # self.fear_traj_lengths = scipy.io.loadmat('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/fear_velocity_traj_v3.mat')['lengths']

        # self.no_fear_traj = scipy.io.loadmat('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/no_fear_velocity_traj_v3.mat')['input_data']
        # self.no_fear_traj_lengths = scipy.io.loadmat('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/no_fear_velocity_traj_v3.mat')['lengths']

        # self.input_data = self.fear_data
        # self.output_data = np.ones(self.fear_data.shape[0])

        # self.define_model()
        
        self.test_acc = 0
        
    def define_model(self):

        with open('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/fear_vel.pkl', 'rb') as f:
            self.fear_traj = pkl.load(f)

        with open('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/bas_vel.pkl', 'rb') as f:
            self.no_fear_traj = pkl.load(f)

        with open('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/val_fear_vel.pkl', 'rb') as f:
            self.val_fear_traj = pkl.load(f)

        with open('/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24/val_bas_vel.pkl', 'rb') as f:
            self.val_no_fear_traj = pkl.load(f)

        self.input_size = self.fear_traj[0].shape[0]

        for i in range(1, len(self.fear_traj)):
            self.input_size = max(self.fear_traj[i].shape[0], self.input_size)

        for i in range(len(self.no_fear_traj)):
            self.input_size = max(self.no_fear_traj[i].shape[0], self.input_size)

        self.input_size = 20
        # np.max(np.concatenate((self.fear_traj_lengths, self.no_fear_traj_lengths)))
        # UNIFORM DATA VIA INTERPOLATION

        self.interpolate()

        # self.show_profiles()

        self.input_data = np.concatenate((self.no_fear_data, self.fear_data), axis=0)
        self.output_data = np.concatenate((np.zeros(self.no_fear_data.shape[0]), np.ones(self.fear_data.shape[0])))


        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.input_data.shape[1:3]),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10 * self.input_size, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # check input dimensionality
        prediction = self.model(self.input_data[:1]).numpy()
        print('test_prediction: {0}'.format(prediction))

        # generate loss function
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        # check output dimensionality and loss function output
        print('test_loss: {0}'.format(self.loss_fn(self.output_data[:1] * (1 + np.random.random([])), prediction).numpy()))

        # assign loss function to model
        self.model.compile(optimizer='adam',
              loss=self.loss_fn,
              metrics=['accuracy'])


        self.model.summary()
     
    def show_profiles(self):
        mean_fear = np.mean(self.fear_data, axis=0)
        std_fear = np.std(self.fear_data, axis=0)

        mean_no_fear = np.mean(self.no_fear_data, axis=0)
        std_no_fear = np.std(self.no_fear_data, axis=0)


        fig = make_subplots(rows=3, cols=1)

        for ax_ind in range(0, 3):
            fig.append_trace(go.Scatter(x=np.linspace(0, self.input_size, self.input_size), y = mean_fear[ax_ind, :] + std_fear[ax_ind, :],
                                    mode='lines', line=dict(color='red',width =0.05), showlegend=False), row=ax_ind + 1, col=1)
            
            fig.append_trace(go.Scatter(x=np.linspace(0, self.input_size, self.input_size), 
                                            y=mean_fear[ax_ind, :], marker=dict(color='red', ), fill='tonexty', showlegend=False), row=ax_ind + 1, col=1)

            fig.append_trace(go.Scatter(x=np.linspace(0, self.input_size, self.input_size), y = mean_fear[ax_ind, :] - std_fear[ax_ind, :], fill='tonexty',
                                    mode='lines', line=dict(color='red',width =0.05), showlegend=False), row=ax_ind + 1, col=1)
            
            fig.append_trace(go.Scatter(x=np.linspace(0, self.input_size, self.input_size), y = mean_no_fear[ax_ind, :] + std_no_fear[ax_ind, :],
                                    mode='lines', line=dict(color='green',width =0.05), showlegend=False), row=ax_ind + 1, col=1)

            fig.append_trace(go.Scatter(x=np.linspace(0, self.input_size, self.input_size), 
                                            y=mean_no_fear[ax_ind, :], marker=dict(color='green', ), fill='tonexty', showlegend=False), row=ax_ind + 1, col=1)

            fig.append_trace(go.Scatter(x=np.linspace(0, self.input_size, self.input_size), y = mean_no_fear[ax_ind, :] - std_no_fear[ax_ind, :], fill='tonexty',
                                    mode='lines', line=dict(color='green',width =0.05), showlegend=False), row=ax_ind + 1, col=1)

        fig.update_layout(
            # title='Fear profile played: {}'.format(exec_ind),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            #font=dict(size=30)
        )

        fig.update_xaxes(showline=True, linecolor='black', gridcolor='black')
        fig.update_yaxes(showline=True, linecolor='black', gridcolor='black')

        fig.update_xaxes(zeroline=True, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinecolor='black')

        fig['layout']['xaxis3']['title']='samples'
        fig['layout']['yaxis']['title']='x - velocity'
        fig['layout']['yaxis2']['title']='y - velocity'
        fig['layout']['yaxis3']['title']='z - velocity'

        fig.write_image("Figures/pdf/MOTION_PRIMITIVE_LIBRARY_V2.pdf")
        fig.show()
        
    def interpolate(self, path='/home/redolaptop/Documents/generative_optimization_trajectory/Training_data/09_02_24'):
        self.fear_data = np.zeros((len(self.fear_traj), self.fear_traj[0].shape[0], self.input_size), dtype=np.float32)
        self.no_fear_data = np.zeros((len(self.no_fear_traj), self.no_fear_traj[0].shape[0], self.input_size), dtype=np.float32)

        for i in range(self.fear_data.shape[0]):
            t_len = self.fear_traj[i].shape[1]
            f_x = interp1d(np.linspace(0, t_len - 1, t_len), self.fear_traj[i][0, 0:t_len], kind = 'cubic')
            f_y = interp1d(np.linspace(0, t_len - 1, t_len), self.fear_traj[i][1, 0:t_len], kind = 'cubic')
            f_z = interp1d(np.linspace(0, t_len - 1, t_len), self.fear_traj[i][2, 0:t_len], kind = 'cubic')
            self.fear_data[i, 0, :] = f_x(np.linspace(0, t_len - 1, self.input_size))
            self.fear_data[i, 1, :] = f_y(np.linspace(0, t_len - 1, self.input_size))
            self.fear_data[i, 2, :] = f_z(np.linspace(0, t_len - 1, self.input_size))

        for i in range(self.no_fear_data.shape[0]):
            t_len = self.no_fear_traj[i].shape[1]
            f_x = interp1d(np.linspace(0, t_len - 1, t_len), self.no_fear_traj[i][0, 0:t_len], kind = 'cubic')
            f_y = interp1d(np.linspace(0, t_len - 1, t_len), self.no_fear_traj[i][1, 0:t_len], kind = 'cubic')
            f_z = interp1d(np.linspace(0, t_len - 1, t_len), self.no_fear_traj[i][2, 0:t_len], kind = 'cubic')
            self.no_fear_data[i, 0, :] = f_x(np.linspace(0, t_len - 1, self.input_size))
            self.no_fear_data[i, 1, :] = f_y(np.linspace(0, t_len - 1, self.input_size))
            self.no_fear_data[i, 2, :] = f_z(np.linspace(0, t_len - 1, self.input_size))
        
        np.save(path + '/fear_data.npy', self.fear_data)
        np.save(path + '/no_fear_data.npy', self.no_fear_data)

        self.val_fear_data = np.zeros((len(self.val_fear_traj), self.val_fear_traj[0].shape[0], self.input_size), dtype=np.float32)
        self.val_no_fear_data = np.zeros((len(self.val_no_fear_traj), self.val_no_fear_traj[0].shape[0], self.input_size), dtype=np.float32)

        for i in range(self.val_fear_data.shape[0]):
            t_len = self.val_fear_traj[i].shape[1]
            f_x = interp1d(np.linspace(0, t_len - 1, t_len), self.val_fear_traj[i][0, 0:t_len], kind = 'cubic')
            f_y = interp1d(np.linspace(0, t_len - 1, t_len), self.val_fear_traj[i][1, 0:t_len], kind = 'cubic')
            f_z = interp1d(np.linspace(0, t_len - 1, t_len), self.val_fear_traj[i][2, 0:t_len], kind = 'cubic')
            self.val_fear_data[i, 0, :] = f_x(np.linspace(0, t_len - 1, self.input_size))
            self.val_fear_data[i, 1, :] = f_y(np.linspace(0, t_len - 1, self.input_size))
            self.val_fear_data[i, 2, :] = f_z(np.linspace(0, t_len - 1, self.input_size))

        for i in range(self.val_no_fear_data.shape[0]):
            t_len = self.val_no_fear_traj[i].shape[1]
            f_x = interp1d(np.linspace(0, t_len - 1, t_len), self.val_no_fear_traj[i][0, 0:t_len], kind = 'cubic')
            f_y = interp1d(np.linspace(0, t_len - 1, t_len), self.val_no_fear_traj[i][1, 0:t_len], kind = 'cubic')
            f_z = interp1d(np.linspace(0, t_len - 1, t_len), self.val_no_fear_traj[i][2, 0:t_len], kind = 'cubic')
            self.val_no_fear_data[i, 0, :] = f_x(np.linspace(0, t_len - 1, self.input_size))
            self.val_no_fear_data[i, 1, :] = f_y(np.linspace(0, t_len - 1, self.input_size))
            self.val_no_fear_data[i, 2, :] = f_z(np.linspace(0, t_len - 1, self.input_size))

        np.save(path + '/val_fear_data.npy', self.val_fear_data)
        np.save(path + '/val_no_fear_data.npy', self.val_no_fear_data)

    def train(self, epochs, batch_size = 10):

        self.define_model()

        training = self.model.fit(self.input_data, self.output_data, epochs=epochs, batch_size=batch_size)

        aver_accuracy = np.zeros(epochs)
        aver = deque(maxlen=20)

        for i in range(epochs):
            aver.append(training.history['accuracy'][i])    
            aver_accuracy[i] = np.mean(aver)

        labels = ['loss', 'accuracy']
        widths = [2, 0.5]

        x_values = np.linspace(0, epochs, epochs)

        fig = make_subplots(rows=len(labels), cols=1)

        for ax_ind in range(0, len(labels)):
            fig.append_trace(go.Scatter(x=x_values, y = training.history[labels[ax_ind]],
                                    mode='lines', line=dict(color='blue', width=widths[ax_ind]), showlegend=False), row=ax_ind + 1, col=1)

        fig.append_trace(go.Scatter(x=x_values, y = aver_accuracy,
                                    mode='lines', line=dict(color='blue', width=2), showlegend=False), row=ax_ind + 1, col=1)

        fig.update_layout(
            # title='Fear profile played: {}'.format(exec_ind),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            #font=dict(size=30)
        )

        fig.update_xaxes(showline=True, linecolor='black', gridcolor='black')
        fig.update_yaxes(showline=True, linecolor='black', gridcolor='black')

        fig.update_xaxes(zeroline=True, zerolinecolor='black')
        fig.update_yaxes(zeroline=True, zerolinecolor='black')

        fig['layout']['xaxis2']['title']='epochs'
        fig['layout']['yaxis']['title']='training loss'
        fig['layout']['yaxis2']['title']='accuracy'

        fig.show()

        return self.model
    
    def cross_validate(self, epochs, sessions=5, batch_size=20):
        loss = []
        accuracy = []
        ma_accuracy = []

        labels = ['loss', 'accuracy']

        for s in tqdm(range(sessions)):
            data_loader.loader(False)

            self.define_model()

            training = self.model.fit(self.input_data, self.output_data, epochs=epochs, batch_size=batch_size, verbose=False)
            loss.append(training.history[labels[0]])
            accuracy.append(training.history[labels[1]])

            self.validate_model()

            aver_accuracy = np.zeros(epochs)
            aver = deque(maxlen=20)

            for i in range(epochs):
                aver.append(training.history['accuracy'][i])    
                aver_accuracy[i] = np.mean(aver)

            aver.clear()

            ma_accuracy.append(aver_accuracy)

        avg_loss = np.mean(loss, axis=0)
        std_loss = np.std(loss, axis=0)

        avg_accuracy = np.mean(accuracy, axis=0)
        std_accuracy = np.std(accuracy, axis=0)

        x_values = np.linspace(0, epochs, epochs)

        fig = make_subplots(rows=2, cols=1)

        fig.append_trace(go.Scatter(x=x_values, y = avg_loss,
                                    mode='lines', line=dict(color='blue', width=2), showlegend=False), row=1, col=1)

        fig.append_trace(go.Scatter(x=x_values, y = avg_accuracy,
                                    mode='lines', line=dict(color='blue', width=2), showlegend=False), row=2, col=1)

        fig.update_layout(
            # title='Fear profile played: {}'.format(exec_ind),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, None]),
            # yaxis2=dict(range=[0, 1]),
            #font=dict(size=30)
        )

        fig.update_xaxes(showline=True, linecolor='black')
        fig.update_yaxes(showline=True, linecolor='black')

        # fig.update_xaxes(showline=True, linecolor='black', gridcolor='grey')
        # fig.update_yaxes(showline=True, linecolor='black', gridcolor='grey')

        # fig.update_xaxes(zeroline=True, zerolinecolor='black')
        # fig.update_yaxes(zeroline=True, zerolinecolor='black')

        # fig['layout']['xaxis2']['title']='epochs'
        # fig['layout']['yaxis']['title']='training loss'
        # fig['layout']['yaxis2']['title']='accuracy'

        fig.write_image("Figures/pdf/TRAINING_RESULTS_MPLV2.pdf")
        fig.show()

    def validate_model(self):
        prediction = self.model(self.fear_data).numpy()
        fear_success_rate = 100*np.sum(prediction > 0.9)/prediction.shape[0]
        print('fear_prediction: {0} pm {1}'.format(np.mean(prediction), np.std(prediction)))
        train_misclass_fear = 100*np.sum(prediction < 0.1)/prediction.shape[0]

        prediction = self.model(self.no_fear_data).numpy()
        no_fear_success_rate = 100*np.sum(prediction < 0.1)/prediction.shape[0]
        print('no_fear_prediction: {0} pm {1}'.format(np.mean(prediction), np.std(prediction)))
        print('success rate: fear {0}%, no fear {1}%'.format(fear_success_rate, no_fear_success_rate))
        train_misclass_no_fear = 100*np.sum(prediction > 0.9)/prediction.shape[0]

        print('training miscalssification: fear {0}, no fear {1}'.format(train_misclass_fear, train_misclass_no_fear))


        prediction = self.model(self.val_fear_data).numpy()
        fear_success_rate = 100*np.sum(prediction > 0.9)/prediction.shape[0]
        print('validation_fear_prediction: {0} pm {1}'.format(np.mean(prediction), np.std(prediction)))
        val_misclass_fear = 100*np.sum(prediction < 0.1)/prediction.shape[0]

        prediction = self.model(self.val_no_fear_data).numpy()
        no_fear_success_rate = 100*np.sum(prediction < 0.1)/prediction.shape[0]
        print('validation_no_fear_prediction: {0} pm {1}'.format(np.mean(prediction), np.std(prediction)))
        print('success rate: fear {0}%, no fear {1}%'.format(fear_success_rate, no_fear_success_rate))
        val_misclass_no_fear = 100*np.sum(prediction > 0.9)/prediction.shape[0]

        print('validation miscalssification: fear {0}, no fear {1}'.format(val_misclass_fear, val_misclass_no_fear))

        

def main():
    print("TensorFlow version:", tf.__version__)

    discriminator = Discriminator()

    data_loader.loader(False)

    model = discriminator.train(epochs=300)

    # model = tf.keras.models.load_model('/home/redolaptop/Documents/generative_optimization_trajectory/Trained_models/discriminator_model.h5')

    discriminator.validate_model()

    model.save('/home/redolaptop/Documents/generative_optimization_trajectory/Trained_models/discriminator_model_1.keras', model)


if __name__ == '__main__':
    discriminator = Discriminator()
    # discriminator.cross_validate(epochs=300)

    main()