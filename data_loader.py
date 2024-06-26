# RAW DATA MUST BE Mx3xT

import scipy.io
import numpy as np
import pickle as pkl

def loader(flag = True):

    # Load the .mat file
    data = scipy.io.loadmat('Training_data/VelocityData_090224.mat')

    factor = 10                                    

    vel_data = data['VelocityData']
    labels = vel_data[0,0].dtype.names                           # ACCESS PARTICIPANT ID
    # print(labels)
    forbidden_labels = ['P008', 'P015', 'P012']
    labels_V0 = labels
    # for i in range(len(labels)):
    #     if labels[i] in forbidden_labels:
    #         labels_V0.append(labels[i])
    # print(labels_V0)
    labels_V1 = vel_data[0,0][labels_V0[0]][0,0].dtype.names        # ACCESS PARTICIPANT DATA IDS
    # print(labels_V1)

    fear_vel = []
    fear_vel_length = []
    fear_vel_ID =[]
    fear_vel_trial_ID =[]

    no_fear_vel = []
    no_fear_vel_length = []
    no_fear_vel_ID =[]
    no_fear_vel_trial_ID =[]

    val_fear_vel = []
    val_fear_vel_length = []
    val_fear_vel_ID =[]
    val_fear_vel_trial_ID =[]

    val_no_fear_vel = []
    val_no_fear_vel_length = []
    val_no_fear_vel_ID =[]
    val_no_fear_vel_trial_ID =[]

    bas_vel = []
    bas_vel_length = []
    bas_vel_ID =[]
    bas_vel_trial_ID =[]

    val_bas_vel = []
    val_bas_vel_length = []
    val_bas_vel_ID =[]
    val_bas_vel_trial_ID =[]

    for id0 in labels_V0:
        user = vel_data[0,0][id0]

        print('USER {}'.format(id0))
        
        for i in range(vel_data[0,0][id0].shape[1]):
            # vel_data[0,0][id0][0, i][labels_V1[15]]
            phase = vel_data[0,0][id0][0, i]['Phase'][0, 0].flatten()
            subphase = vel_data[0,0][id0][0, i]['Subphase'][0, 0].flatten()
            condition = vel_data[0,0][id0][0, i]['Condition'][0, 0].flatten()                                       # 1 is fear 0 is no fear
            steps = vel_data[0,0][id0][0, i]['Movement_Time'][0, 0].flatten()
            # type = vel_data[0,0][id0][0, i]['Phenotype'][0, 0].flatten()
            if np.random.rand() > 0.7: validation = 1
            else: validation = 0
            if (phase == 'ASSOCIATION' and (subphase == 3)) or (phase == 'EXTINCTION' and (subphase == 1)):
                if validation == 0:
                    if condition == 1:
                        # print('TRIAL {} PHASE {} SUBPHASE {} CONDITION {}'.format(i, phase, subphase, condition))
                        fear_vel.append((vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].T).astype(np.float32))
                        fear_vel_length.append(vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].shape[0])
                        fear_vel_ID.append(id0)
                        fear_vel_trial_ID.append(vel_data[0,0][id0][0, i]['Trial_Number'][0, 0])
                    else: 
                        no_fear_vel.append((vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].T).astype(np.float32))
                        no_fear_vel_length.append(vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].shape[0])
                        no_fear_vel_ID.append(id0)
                        no_fear_vel_trial_ID.append(vel_data[0,0][id0][0, i]['Trial_Number'][0, 0])
                else:
                    if condition == 1:
                        # print('TRIAL {} PHASE {} SUBPHASE {} CONDITION {}'.format(i, phase, subphase, condition))
                        val_fear_vel.append((vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].T).astype(np.float32))
                        val_fear_vel_length.append(vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].shape[0])
                        val_fear_vel_ID.append(id0)
                        val_fear_vel_trial_ID.append(vel_data[0,0][id0][0, i]['Trial_Number'][0, 0])
                    else:
                        val_no_fear_vel.append((vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].T).astype(np.float32))
                        val_no_fear_vel_length.append(vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].shape[0])
                        val_no_fear_vel_ID.append(id0)
                        val_no_fear_vel_trial_ID.append(vel_data[0,0][id0][0, i]['Trial_Number'][0, 0])
            if phase == 'BASELINE':
                if validation == 0:
                    # print('TRIAL {} PHASE {} SUBPHASE {} CONDITION {}'.format(i, phase, subphase, condition))
                    bas_vel.append((vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].T).astype(np.float32))
                    bas_vel_length.append(vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].shape[0])
                    bas_vel_ID.append(id0)
                    bas_vel_trial_ID.append(vel_data[0,0][id0][0, i]['Trial_Number'][0, 0])
                else:
                    # print('TRIAL {} PHASE {} SUBPHASE {} CONDITION {}'.format(i, phase, subphase, condition))
                    val_bas_vel.append((vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].T).astype(np.float32))
                    val_bas_vel_length.append(vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].shape[0])
                    val_bas_vel_ID.append(id0)
                    val_bas_vel_trial_ID.append(vel_data[0,0][id0][0, i]['Trial_Number'][0, 0])
            '''
                else:
                    print('TRIAL {} PHASE {} SUBPHASE {} CONDITION {}'.format(i, phase, subphase, condition))
                    val_no_fear_vel.append((vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].T).astype(np.float32))
                    val_no_fear_vel_length.append(vel_data[0,0][id0][0, i]['Vxyzf'][0, 0].shape[0])
                    val_no_fear_vel_ID.append(id0)
                    val_no_fear_vel_trial_ID.append(vel_data[0,0][id0][0, i]['Trial_Number'][0, 0])
            '''
                    
    print('FEAR SAMPLES {}, NO FEAR SAMPLES {} | BASELINE SAMPLES {}'.format(len(fear_vel), len(no_fear_vel), len(bas_vel)))    

    print('VALIDATION FEAR SAMPLES {}, VALIDATION NO FEAR SAMPLES {}| VALIDATION BASELINE SAMPLES {}'.format(len(val_fear_vel), len(val_no_fear_vel), len(val_bas_vel)))  

    print('MAX LENGHT {}'.format(max(np.max(fear_vel_length), np.max(bas_vel_length), np.max(val_fear_vel_length), np.max(val_bas_vel_length))))

    print('FEAR SAMPLES {} | NO FEAR SAMPLES {}'.format(len(fear_vel_length) + len(val_fear_vel_length), len(bas_vel_length) + len(val_bas_vel_length)))

    if flag:  
        with open('Training_data/fear_vel.pkl', 'wb') as f:
            pkl.dump(fear_vel, f)
        f.close()

        with open('Training_data/no_fear_vel.pkl', 'wb') as f:
            pkl.dump(no_fear_vel, f)
        f.close()
        
        with open('Training_data/bas_vel.pkl', 'wb') as f:
            pkl.dump(bas_vel, f)
        f.close()

        with open('Training_data/val_fear_vel.pkl', 'wb') as f:
            pkl.dump(val_fear_vel, f)
        f.close()

        with open('Training_data/val_no_fear_vel.pkl', 'wb') as f:
            pkl.dump(val_no_fear_vel, f)
        f.close()

        with open('Training_data/val_bas_vel.pkl', 'wb') as f:
            pkl.dump(val_bas_vel, f)
        f.close()

        np.save('Training_data/fear_vel_len.npy', fear_vel_length)
        np.save('Training_data/no_fear_vel_len.npy', no_fear_vel_length)
        np.save('Training_data/bas_vel_len.npy', bas_vel_length)

        np.save('Training_data/fear_vel_ID.npy', fear_vel_ID)
        np.save('Training_data/no_fear_vel_ID.npy', no_fear_vel_ID)
        np.save('Training_data/bas_vel_ID.npy', bas_vel_ID)

        np.save('Training_data/fear_vel_trial_ID.npy', fear_vel_trial_ID)
        np.save('Training_data/no_fear_vel_trial_ID.npy', no_fear_vel_trial_ID)
        np.save('Training_data/bas_vel_trial_ID.npy', bas_vel_trial_ID)

        np.save('Training_data/val_fear_vel_len.npy', val_fear_vel_length)
        np.save('Training_data/val_no_fear_vel_len.npy', val_no_fear_vel_length)
        np.save('Training_data/val_bas_vel_len.npy', val_bas_vel_length)

        np.save('Training_data/val_fear_vel_ID.npy', val_fear_vel_ID)
        np.save('Training_data/val_no_fear_vel_ID.npy', val_no_fear_vel_ID)
        np.save('Training_data/val_bas_vel_ID.npy', val_bas_vel_ID)

        np.save('Training_data/val_fear_vel_trial_ID.npy', val_fear_vel_trial_ID)
        np.save('Training_data/val_no_fear_vel_trial_ID.npy', val_no_fear_vel_trial_ID)
        np.save('Training_data/val_bas_vel_trial_ID.npy', val_bas_vel_trial_ID)


if __name__ == '__main__':
    loader()

