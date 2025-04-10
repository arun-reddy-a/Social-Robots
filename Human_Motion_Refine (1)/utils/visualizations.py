import numpy as np
import pickle
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from smplx import SMPLX

# Function to plot the skeleton
def plot_skeleton(joints, ax, c, joint_names):
    # Define the connections between joints
    skeleton_connections = [
        ('pelvis', 'spine1'),('pelvis', 'right_hip'), ('pelvis','left_hip'), ('spine1', 'spine2'), ('spine2', 'spine3'), ('spine3', 'neck'),
        ('neck', 'head'), ('spine3', 'left_collar'), ('spine3', 'right_collar'),
        ('left_collar', 'left_shoulder'), ('right_collar', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
        ('left_ankle', 'left_foot'), ('right_ankle', 'right_foot'),
        ('head', 'jaw'), ('jaw', 'left_eye_smplhf'), ('jaw', 'right_eye_smplhf'),
        ('left_wrist', 'left_thumb1'), ('left_wrist', 'left_index1'), ('left_wrist', 'left_middle1'),
        ('left_wrist', 'left_ring1'), ('left_wrist', 'left_pinky1'),
        ('right_wrist', 'right_thumb1'), ('right_wrist', 'right_index1'), ('right_wrist', 'right_middle1'),
        ('right_wrist', 'right_ring1'), ('right_wrist', 'right_pinky1'), ('left_thumb1', 'left_thumb2'),
        ('left_index1', 'left_index2'), ('left_middle1', 'left_middle2'), ('left_ring1', 'left_ring2'),
        ('left_pinky1', 'left_pinky2'), ('right_thumb1', 'right_thumb2'), ('right_index1', 'right_index2'),
        ('right_middle1', 'right_middle2'), ('right_ring1', 'right_ring2'), ('right_pinky1', 'right_pinky2'),
        ('left_thumb2', 'left_thumb3'), ('left_index2', 'left_index3'), ('left_middle2', 'left_middle3'),
        ('left_ring2', 'left_ring3'), ('left_pinky2', 'left_pinky3'), ('right_thumb2', 'right_thumb3'),
        ('right_index2', 'right_index3'), ('right_middle2', 'right_middle3'), ('right_ring2', 'right_ring3'),
        ('right_pinky2', 'right_pinky3')
    ]

    # Convert joint names to indices
    joint_indices = {name: i for i, name in enumerate(joint_names)}

    for joint1_name, joint2_name in skeleton_connections:
        joint1_index = joint_indices[joint1_name]
        joint2_index = joint_indices[joint2_name]

        joint1 = joints[joint1_index]
        joint2 = joints[joint2_index]

        ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], color=c)

def vis_one_data_point(leader, follower, coords_details):
    ''' leader and follower should have shapes (T, 165) '''
    ''' will visualize a single data_point '''  
    # Example data
    T = leader.shape[0]  # Number of frames

    # if joints_data is a tensor, convert
    if torch.is_tensor(leader):
        joints_data = leader.reshape(-1, 55, 3).detach().cpu().numpy()
        joints_data_cond = follower.reshape(-1, 55, 3).detach().cpu().numpy()
    else:
        joints_data = leader.reshape(-1, 55, 3)
        joints_data_cond = follower.reshape(-1, 55, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # List of joint names
    joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2',
        'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
        'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2',
        'left_ring3', 'left_thumb1', 'left_thumb2', 'left_thumb3', 'right_index1',
        'right_index2', 'right_index3', 'right_middle1', 'right_middle2',
        'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3',
        'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1',
        'right_thumb2', 'right_thumb3'
    ]

    def plot_frame(t):
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        
        # Plot joints
        ax.scatter(joints_data[t, :, 0], joints_data[t, :, 1], joints_data[t, :, 2], c='r')
        ax.scatter(joints_data_cond[t, :, 0], joints_data_cond[t, :, 1], joints_data_cond[t, :, 2], c='b')
        
        # Plot skeleton
        plot_skeleton(joints_data[t], ax, 'r', joint_names)
        plot_skeleton(joints_data_cond[t], ax, 'b', joint_names)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Skeleton Visualization - Frame {t+1}')

        '''xmin = np.min(joints_data[:, :, 0])
        xmin = min(xmin, np.min(joints_data_cond[:, :, 0]))
        xmax = np.max(joints_data[:, :, 0])
        xmax = max(xmax, np.max(joints_data_cond[:, :, 0]))
        ymin = np.min(joints_data[:, :, 1])
        ymin = min(ymin, np.min(joints_data_cond[:, :, 1]))
        ymax = np.max(joints_data[:, :, 1])
        ymax = max(ymax, np.max(joints_data_cond[:, :, 1]))
        zmin = np.min(joints_data[:, :, 2])
        zmin = min(zmin, np.min(joints_data_cond[:, :, 2]))
        zmax = np.max(joints_data[:, :, 2])
        zmax = max(zmax, np.max(joints_data_cond[:, :, 2]))'''

        xmin = coords_details['xmin']
        xmax = coords_details['xmax']
        ymin = coords_details['ymin']
        ymax = coords_details['ymax']
        zmin = coords_details['zmin']
        zmax = coords_details['zmax']

        # Set fixed axis limits
        ax.set_xlim([xmin, xmax])  # Adjust xmin and xmax according to your data
        ax.set_ylim([ymin, ymax])  # Adjust ymin and ymax according to your data
        ax.set_zlim([zmin, zmax])  # Adjust zmin and zmax according to your data



    # Function to update the plot for the animation
    def update(t):
        ax.clear()  # Clear the current figure
        plot_frame(t)

    ani = animation.FuncAnimation(fig, update, frames=range(leader.shape[0]), repeat=False)

    plt.close(fig)
    return ani
    #ipywidgets.interact(plot_frame, t=ipywidgets.Play(min=0, max=leader.shape[0]-1, step=1, interval=20))

def smplx_to_pos3d(data):

    smplx = SMPLX(model_path='./smplx', betas=data[:, 171:187][:, :10], gender='male', \
        batch_size=len(data[:, 171:187]), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)
    
    
    keypoints3d = smplx.forward(
        global_orient=torch.from_numpy(data[:, 3:6]).float(),
        body_pose=torch.from_numpy(data[:, 6:171][:, 3:66]).float(),
        jaw_pose=torch.from_numpy(data[:, 6:171][:, 66:69]).float(),
        leye_pose=torch.from_numpy(data[:, 6:171][:, 69:72]).float(),
        reye_pose=torch.from_numpy(data[:, 6:171][:, 72:75]).float(),
        left_hand_pose=torch.from_numpy(data[:, 6:171][:, 75:120]).float(),
        right_hand_pose=torch.from_numpy(data[:, 6:171][:, 120:]).float(),
        transl=torch.from_numpy(data[:, :3]).float(),
        betas=torch.from_numpy(data[:, 171:187][:, :10]).float()
        ).joints.detach().numpy()[:, :55]

    nframes = keypoints3d.shape[0]
    return keypoints3d.reshape(nframes, -1)

def smplx_to_pos3d_torch(data):

    smplx = SMPLX(model_path='./smplx', betas=data[:, 171:187][:, :10], gender='male', \
        batch_size=len(data[:, 171:187]), num_betas=10, use_pca=False, use_face_contour=True, flat_hand_mean=True)
    
    smplx = smplx.to('cuda:0')
    keypoints3d = smplx.forward(
        global_orient=(data[:, 3:6]).float(),
        body_pose=(data[:, 6:171][:, 3:66]).float(),
        jaw_pose=(data[:, 6:171][:, 66:69]).float(),
        leye_pose=(data[:, 6:171][:, 69:72]).float(),
        reye_pose=(data[:, 6:171][:, 72:75]).float(),
        left_hand_pose=(data[:, 6:171][:, 75:120]).float(),
        right_hand_pose=(data[:, 6:171][:, 120:]).float(),
        transl=(data[:, :3]).float(),
        betas=(data[:, 171:187][:, :10]).float()
        ).joints[:, :55]

    nframes = keypoints3d.shape[0]
    return keypoints3d.reshape(nframes, -1)

def vis_results (input, ground_truth, predicted, epoch, path, is_smplx=False):
    '''
    input: (N, 300, 165)
    ground_truth: (N, 300, 165)
    predicted: (N, 300, 165)
    '''
    time_steps = input.shape[1]
    input_ = input.detach().cpu().numpy()
    ground_truth_ = ground_truth.detach().cpu().numpy()
    predicted_ = predicted.detach().cpu().numpy()
    if is_smplx:
        # Convert to pos_3d before plotting
        input_ = np.array([smplx_to_pos3d(data) for data in input_])
        ground_truth_ = np.array([smplx_to_pos3d(data) for data in ground_truth_])
        predicted_ = np.array([smplx_to_pos3d(data) for data in predicted_])
    num_samples = input_.shape[0]
    size = min(5, input_.shape[0])
    random_indices = np.random.choice(input_.shape[0], size=size, replace=False)
    input_ = input_[random_indices]
    ground_truth_ = ground_truth_[random_indices]
    predicted_ = predicted_[random_indices]
    plot_num_samples = input_.shape[0]

    data_concat = np.concatenate([input_, ground_truth_, predicted_], axis=0).reshape(-1, time_steps, 55, 3)

    xmin = np.min(data_concat[:, :, :, 0])
    xmax = np.max(data_concat[:, :, :, 0])

    ymin = np.min(data_concat[:, :, :, 1])
    ymax = np.max(data_concat[:, :, :, 1])

    zmin = np.min(data_concat[:, :, :, 2])
    zmax = np.max(data_concat[:, :, :, 2])

    coords_details = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'zmin': zmin, 'zmax': zmax}
    
    print(f'Plotting at epoch {epoch}')
    for i in tqdm(range(plot_num_samples)):
        ani_gt = vis_one_data_point(input_[i], ground_truth_[i], coords_details)
        ani_pred = vis_one_data_point(input_[i], predicted_[i], coords_details)
        ani_gt.save(path+f'gt_epoch_{epoch}_{i}.gif', writer=animation.PillowWriter(fps=15))
        ani_pred.save(path+f'pred_epoch_{epoch}_{i}.gif', writer=animation.PillowWriter(fps=15))

    return None


def count_parameters(model: nn.Module):
    """
    Count the total number of parameters and the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        total_params (int): The total number of parameters in the model.
        trainable_params (int): The number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params