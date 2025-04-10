import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from timeit import default_timer
import torch.nn.functional as F
import random
from utils.gaussian_random_field import *
from utils.visualizations import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import gc

train_mean = torch.from_numpy(np.load('data/data_more_processing/train_mean.npy')).to('cuda:0')
train_std = torch.from_numpy(np.load('data/data_more_processing/train_std.npy')).to('cuda:0')

writer = SummaryWriter("/scratch/gilbreth/anugu/logs_demo/demo_2_models_overfit_one_point_v10")

def calculate_gradient_penalty(model, real_images,c, fake_images,cf, device, res):
    """Calculates the gradient penalty loss for WGAN GRF"""
    # Random weight term for interpolation between real and fake data
    alpha = torch.randn((real_images.size(0),1, 1), device=device)
    # Get random interpolation between real and fake data
    interpolates = (alpha * real_images + ((1 - alpha) * fake_images)).requires_grad_(True)
    #print(real_images.shape,c.shape,fake_images.shape, interpolates.shape)

    model_interpolates = model(interpolates.float(),c.requires_grad_(True))
    grad_outputs = torch.ones(model_interpolates.size(), device=device, requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=model_interpolates,
        inputs=(interpolates,c),
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )

    gradients_x = gradients[0].reshape(gradients[0].shape[0], -1)
    gradients_c = gradients[1].reshape(gradients[1].shape[0], -1)

    gradient_penalty = torch.mean(((gradients_x.norm(2, dim=1)**2+gradients_c.norm(2, dim=1)**2)**0.5 - 1/np.sqrt(res*res)) ** 2) 
    return gradient_penalty

def convert_to_axis_angle(data):
    '''data [T, 242]'''
    axis_magnitude = data[:, 6:226].reshape(-1, 55, 4)
    axis = axis_magnitude[:, :, :3]
    magnitude = axis_magnitude[:, :, 3:]
    poses = axis * (magnitude+1e-10)
    poses = poses.reshape(axis_magnitude.shape[0], -1)
    new_data = torch.cat([data[:, :6], poses, data[:, 226:]], axis=1)
    return new_data

def human_loss_3d(data):
    ''' data is of shape (N, T, J, 3) '''
    right = [(0, 2),(9, 14),(14, 17),(17, 19),(19, 21),(2, 5),(5, 8),(8, 11),(22, 24)
            ,(21, 52),(21, 40),(21, 43),(21, 49),(21, 46),(52, 53),(40, 41),(43, 44),
            (49, 50),(46, 47),(53, 54),(41, 42),(44, 45),(50, 51),(47, 48)]

    left = [(0, 1), (9, 13),(13, 16),(16, 18),(18, 20),(1, 4),(4, 7),(7, 10),
            (22, 23),(20, 37),(20, 25),(20, 28),(20, 34),(20, 31),(37, 38),
            (25, 26),(28, 29),(34, 35),(31, 32),(38, 39),(26, 27),(29, 30),
            (35, 36),(32, 33)]

    error = 0

    for i in range(len(right)):
        error+=(torch.norm(data[:,:,right[i][0],:] - data[:,:,right[i][1],:])\
             - torch.norm(data[:,:,left[i][0],:] - data[:,:,left[i][1],:]))**2
    return error

def unnormalize_data(data, mean=train_mean, std=train_std):
    ''' data: [T, 187]'''
    return data * (std+1e-10) + mean

def human_loss(data):
    ''' data is of (N, T, 242) tensor, in normalized axis-magnitude form'''
    axis_angle_data = torch.stack([convert_to_axis_angle(point) for point in data])
    unnormalized_axis_angle_data = unnormalize_data(axis_angle_data)
    pos_3d_data = torch.stack([smplx_to_pos3d_torch(point) for point in unnormalized_axis_angle_data]).reshape(axis_angle_data.shape[0], axis_angle_data.shape[1], 55, 3)
    return human_loss_3d(pos_3d_data)

def foot_contact_loss(data):
    ''' data is of (N, T, 242) tensor, in normalized axis-magnitude form'''
    axis_angle_data = torch.stack([convert_to_axis_angle(point) for point in data])
    unnormalized_axis_angle_data = unnormalize_data(axis_angle_data)
    pos_3d_data = torch.stack([smplx_to_pos3d_torch(point) for point in unnormalized_axis_angle_data]).reshape(axis_angle_data.shape[0], axis_angle_data.shape[1], 55, 3) # (N, T, 55, 3)
    left_foot = pos_3d_data[:, :, 10, 2] # (T, 3)
    right_foot = pos_3d_data[:, :, 11, 2] # (T, 3)
    loss = ((left_foot-0)**2 + (right_foot-0)**2).mean()
    return loss
    
def balance_loss(input, generated):
    '''input: (N, T, 242), generated: (N, T, 242)'''
    axis_angle_input = torch.stack([convert_to_axis_angle(point) for point in input])
    axis_angle_generated = torch.stack([convert_to_axis_angle(point) for point in generated])

    unnormalized_axis_angle_input = unnormalize_data(axis_angle_input)
    unnormalized_axis_angle_generated = unnormalize_data(axis_angle_generated)

    pos_3d_input = torch.stack([smplx_to_pos3d_torch(point) for point in unnormalized_axis_angle_input]).reshape(axis_angle_input.shape[0], axis_angle_input.shape[1], 55, 3)
    pos_3d_generated = torch.stack([smplx_to_pos3d_torch(point) for point in unnormalized_axis_angle_generated]).reshape(axis_angle_generated.shape[0], axis_angle_generated.shape[1], 55, 3)

    left_foot_input = pos_3d_input[:, :, 10, 2] # (N, T)
    right_foot_input = pos_3d_input[:, :, 11, 2] # (N, T)

    ground_level = min(left_foot_input.min(), right_foot_input.min())

    left_foot_generated = pos_3d_generated[:, :, 10, 2] # (N, T)
    right_foot_generated = pos_3d_generated[:, :, 11, 2] # (N, T)

    contact_loss = ((left_foot_generated-ground_level)**2 + (right_foot_generated-ground_level)**2).mean()

    pos_3d_generated_pelvis = pos_3d_generated[:, :, 0, :].reshape(-1, 3) # (N*T, 3)
    pos_3d_generated_neck = pos_3d_generated[:, :, 12, :].reshape(-1, 3) # (N*T, 3)
    magnitude = torch.linalg.norm(pos_3d_generated_neck-pos_3d_generated_pelvis, dim=1) # (N*T, 1)
    difference = pos_3d_generated_neck[:, 2]- pos_3d_generated_pelvis[:, 2] # (N*T, 1)
    dot_product = difference/magnitude # (N*T, 1)

    dot_product_loss = ((dot_product-1)**2).mean()

    return contact_loss + dot_product_loss

def train_GANO_HM_two_stage_copy_copy(D_transl, D_pose, G_transl, G_pose, train_dataloader, val_dataloader,
                                         epochs, D_transl_optimizer, D_pose_optimizer, 
                                         G_transl_optimizer, G_pose_optimizer, λ_grad, grf, n_critic, device,
                                         batch_size, logging, hm_weight, fc_weight, gt_weight, D_transl_scheduler, D_pose_scheduler, G_transl_scheduler, G_pose_scheduler, 
                                         data_stats = None, scheduler=None):
    losses_D_transl = np.zeros(epochs)
    losses_G_transl = np.zeros(epochs)
    losses_W_transl = np.zeros(epochs)
    losses_MSE_transl = np.zeros(epochs)
    losses_V_MSE_transl = np.zeros(epochs)
    no_of_batches = len(train_dataloader)

    losses_D_poses = np.zeros(epochs)
    losses_G_poses = np.zeros(epochs)
    losses_W_poses = np.zeros(epochs)
    losses_MSE_poses = np.zeros(epochs)
    losses_V_MSE_poses = np.zeros(epochs)

    losses_MSE_full = np.zeros(epochs)
    losses_V_MSE_full = np.zeros(epochs)

    for i in tqdm(range(epochs)):
        loss_D_transl = 0.0
        loss_G_transl = 0.0
        loss_W_transl = 0.0
        loss_MSE_transl = 0.0
        loss_V_MSE_transl = 0.0

        loss_D_poses = 0.0
        loss_G_poses = 0.0
        loss_W_poses = 0.0
        loss_MSE_poses = 0.0
        loss_V_MSE_poses = 0.0

        loss_MSE_full = 0.0
        loss_V_MSE_full = 0.0

        for j, data in tqdm(enumerate(train_dataloader)):

            #-------- Training D1 (D_transl)--------#
            
            if random.random()<0.5:
                x = data[0].to(device).float()
                c = data[1].to(device).float()

                x_transl = torch.cat([x[:, :, :3], x[:, :, 226:242]], axis=2)
                c_transl = torch.cat([c[:, :, :3], c[:, :, 226:242]], axis=2)

                x_pose = x[:, :, 3:226]
                c_pose = c[:, :, 3:226]
            else:
                x = data[1].to(device).float()
                c = data[0].to(device).float()

                x_transl = torch.cat([x[:, :, :3], x[:, :, 226:242]], axis=2)
                c_transl = torch.cat([c[:, :, :3], c[:, :, 226:242]], axis=2)

                x_pose = x[:, :, 3:226]
                c_pose = c[:, :, 3:226]

            x_transl = x_transl.reshape(x_transl.shape[0], x_transl.shape[1], -1)
            c_transl = c_transl.reshape(c_transl.shape[0], c_transl.shape[1], -1)

            '''res_transl = x_transl.shape[1] #asuming uniform resolution along x and y
            D_transl_optimizer.zero_grad()
            
            x_syn_transl = G_transl(grf.sample(x_transl.shape[0], x_transl.shape[1]).float(),c_transl)

            W_loss_transl = -torch.mean(D_transl(x_transl,c_transl)) + torch.mean(D_transl(x_syn_transl.detach(),c_transl))

            gradient_penalty_transl = calculate_gradient_penalty(D_transl, x_transl.data,c_transl, x_syn_transl.data,c_transl, device, res_transl)

            loss_D1 = W_loss_transl + λ_grad * gradient_penalty_transl
            loss_D1.backward()

            loss_D_transl += loss_D1.item()
            loss_W_transl += W_loss_transl.item()

            D_transl_optimizer.step()
            D_transl_scheduler.step()
            
            #-------- Training D2 (D_poses)--------#

            x_pose = x_pose.reshape(x_pose.shape[0], x_pose.shape[1], -1)
            c_pose = c_pose.reshape(c_pose.shape[0], c_pose.shape[1], -1)

            res_pose = x_pose.shape[1] #asuming uniform resolution along x and y
            D_pose_optimizer.zero_grad()
            
            
            x_syn_poses = G_pose(grf.sample(x_pose.shape[0], x_pose.shape[1]).float(),c_pose)

            W_loss_pose = -torch.mean(D_pose(x_pose,c_pose)) + torch.mean(D_pose(x_syn_poses.detach(),c_pose))

            gradient_penalty_pose = calculate_gradient_penalty(D_pose, x_pose.data,c_pose, x_syn_poses.data,c_pose, device, res_pose)

            loss_D2 = W_loss_pose + λ_grad * gradient_penalty_pose
            loss_D2.backward()

            loss_D_poses += loss_D2.item()
            loss_W_poses += W_loss_pose.item()

            D_pose_optimizer.step()
            D_pose_scheduler.step()

            x_syn_full = torch.cat([x_syn_transl[:, :, :3], x_syn_poses, x_syn_transl[:, :, 3:]], axis=2)'''
            
            # Train Generators.
            if (j + 1) % n_critic == 0:
                #-------- Training G1 and G2 (G_transl, G_pose)--------#
                '''if random.random()<0.5:
                    x = data[0].to(device).float()
                    c = data[1].to(device).float()

                    x_transl = torch.cat([x[:, :, :3], x[:, :, 226:242]], axis=2)
                    c_transl = torch.cat([c[:, :, :3], c[:, :, 226:242]], axis=2)

                    x_pose = x[:, :, 3:226]
                    c_pose = c[:, :, 3:226]
                else:
                    x = data[1].to(device).float()
                    c = data[0].to(device).float()

                    x_transl = torch.cat([x[:, :, :3], x[:, :, 226:242]], axis=2)
                    c_transl = torch.cat([c[:, :, :3], c[:, :, 226:242]], axis=2)

                    x_pose = x[:, :, 3:226]
                    c_pose = c[:, :, 3:226]

                x_transl = x_transl.reshape(x_transl.shape[0], x_transl.shape[1], -1)
                c_transl = c_transl.reshape(c_transl.shape[0], c_transl.shape[1], -1)

                x_pose = x_pose.reshape(x_pose.shape[0], x_pose.shape[1], -1)
                c_pose = c_pose.reshape(c_pose.shape[0], c_pose.shape[1], -1)'''

                G_transl_optimizer.zero_grad()
                G_pose_optimizer.zero_grad()

                x_syn_transl = G_transl(grf.sample(x_transl.shape[0], x_transl.shape[1]).float(),c_transl)
                x_syn_poses = G_pose(grf.sample(x_pose.shape[0], x_pose.shape[1]).float(),c_pose)
                x_syn_full = torch.cat([x_syn_transl[:, :, :3], x_syn_poses, x_syn_transl[:, :, 3:]], axis=2)

                
                loss_G1 = gt_weight*F.mse_loss(x_syn_transl, x_transl)
                loss_G2 = gt_weight*F.mse_loss(x_syn_poses, x_pose)

                loss_G1.backward(retain_graph=True)
                loss_G2.backward()

                loss_G_transl += loss_G1.item()
                loss_G_poses += loss_G2.item()

                G_transl_optimizer.step()
                G_transl_scheduler.step()
                G_pose_optimizer.step()
                G_pose_scheduler.step()

            
            loss_MSE_transl += F.mse_loss(x_syn_transl, x_transl)  
            loss_MSE_poses += F.mse_loss(x_syn_poses,x_pose)
            loss_MSE_full += (F.mse_loss(x_syn_transl, x_transl)  + F.mse_loss(x_syn_poses,x_pose))/2

            del x, c, x_transl, c_transl, x_pose, c_pose, x_syn_transl, x_syn_poses, x_syn_full
            gc.collect()
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            no_of_val_batches = len(val_dataloader)
            for j, data in enumerate(val_dataloader):
                x = data[0].to(device).float()
                c = data[1].to(device).float()

                x_transl = torch.cat([x[:, :, :3], x[:, :, 226:242]], axis=2)
                c_transl = torch.cat([c[:, :, :3], c[:, :, 226:242]], axis=2)

                x_pose = x[:, :, 3:226]
                c_pose = c[:, :, 3:226]

                x_transl = x_transl.reshape(x_transl.shape[0], x_transl.shape[1], -1)
                c_transl = c_transl.reshape(c_transl.shape[0], c_transl.shape[1], -1)

                x_pose = x_pose.reshape(x_pose.shape[0], x_pose.shape[1], -1)
                c_pose = c_pose.reshape(c_pose.shape[0], c_pose.shape[1], -1)

                x_syn_transl = x_syn_transl = G_transl(grf.sample(x_transl.shape[0], x_transl.shape[1]).float(),c_transl)
                x_syn_poses = G_pose(grf.sample(x_pose.shape[0], x_pose.shape[1]).float(),c_pose)
                x_syn_full = torch.cat([x_syn_transl[:, :, :3], x_syn_poses, x_syn_transl[:, :, 3:]], axis=2)

                loss_V_MSE_transl += F.mse_loss(x_syn_transl, x_transl)
                loss_V_MSE_poses += F.mse_loss(x_syn_poses,x_pose)
                loss_V_MSE_full += (F.mse_loss(x_syn_transl, x_transl)  + F.mse_loss(x_syn_poses,x_pose))/2

                del x, c, x_transl, c_transl, x_pose, c_pose, x_syn_transl, x_syn_poses, x_syn_full
                gc.collect()
                torch.cuda.empty_cache()
        
        
        losses_G_transl[i] = loss_G_transl / (no_of_batches/n_critic)
        losses_MSE_transl[i] = loss_MSE_transl/ no_of_batches
        losses_V_MSE_transl[i] =  loss_V_MSE_transl/ no_of_val_batches

        losses_G_poses[i] = loss_G_poses / (no_of_batches/n_critic)
        losses_MSE_poses[i] = loss_MSE_poses/ no_of_batches
        losses_V_MSE_poses[i] =  loss_V_MSE_poses/ no_of_val_batches

        losses_MSE_full[i] = loss_MSE_full/ no_of_batches
        losses_V_MSE_full[i] = loss_V_MSE_full/ no_of_val_batches

        # Add to tensorboard.
        writer.add_scalar('losses_G_transl', losses_G_transl[i], i)
        writer.add_scalar('losses_MSE_transl', losses_MSE_transl[i], i)
        writer.add_scalar('losses_V_MSE_transl', losses_V_MSE_transl[i], i)

        writer.add_scalar('losses_G_poses', losses_G_poses[i], i)
        writer.add_scalar('losses_MSE_poses', losses_MSE_poses[i], i)
        writer.add_scalar('losses_V_MSE_poses', losses_V_MSE_poses[i], i)

        writer.add_scalar('loss_MSE_full', losses_MSE_full[i], i)
        writer.add_scalar('loss_V_MSE_full', losses_V_MSE_full[i], i)


        if i%100==0:    
            torch.save(G_transl.state_dict(), logging['G_trans_save_path']+f"generator_trans_epoch{i}.pt")

            torch.save(G_pose.state_dict(), logging['G_pose_save_path']+f"generator_pose_epoch{i}.pt")

            torch.save(G_transl_optimizer.state_dict(), logging['G_transl_optimizer_save_path']+f"epoch{i}.pt")

            torch.save(G_pose_optimizer.state_dict(), logging['G_pose_optimizer_save_path']+f"epoch{i}.pt")

            torch.save(torch.from_numpy(losses_G_transl), logging['plots']+f"loss_G_transl.pt")
            torch.save(torch.from_numpy(losses_MSE_transl), logging['plots']+f"loss_MSE_transl.pt")
            torch.save(torch.from_numpy(losses_V_MSE_transl), logging['plots']+f"loss_V_MSE_transl.pt")

            torch.save(torch.from_numpy(losses_G_poses), logging['plots']+f"loss_G_poses.pt")
            torch.save(torch.from_numpy(losses_MSE_poses), logging['plots']+f"loss_MSE_poses.pt")
            torch.save(torch.from_numpy(losses_V_MSE_poses), logging['plots']+f"loss_V_MSE_poses.pt")

            torch.save(torch.from_numpy(losses_MSE_full), logging['plots']+f"loss_MSE_full.pt")
            torch.save(torch.from_numpy(losses_V_MSE_full), logging['plots']+f"loss_V_MSE_full.pt")



        if i%100==0:
            with torch.no_grad():
                for j, data in enumerate(val_dataloader):

                    x = data[0].to(device).float()
                    c = data[1].to(device).float()

                    x_transl = torch.cat([x[:, :, :3], x[:, :, 226:242]], axis=2)
                    c_transl = torch.cat([c[:, :, :3], c[:, :, 226:242]], axis=2)

                    x_pose = x[:, :, 3:226]
                    c_pose = c[:, :, 3:226]

                    x_transl = x_transl.reshape(x_transl.shape[0], x_transl.shape[1], -1)
                    c_transl = c_transl.reshape(c_transl.shape[0], c_transl.shape[1], -1)

                    x_pose = x_pose.reshape(x_pose.shape[0], x_pose.shape[1], -1)
                    c_pose = c_pose.reshape(c_pose.shape[0], c_pose.shape[1], -1)

                    x_syn_transl = G_transl(grf.sample(x_transl.shape[0], x_transl.shape[1]).float(),c_transl)
                    x_syn_poses = G_pose(grf.sample(x_pose.shape[0], x_pose.shape[1]).float(),c_pose)
                    x_syn_full = torch.cat([x_syn_transl[:, :, :3], x_syn_poses, x_syn_transl[:, :, 3:]], axis=2)

                    x_axis_angle = torch.stack([convert_to_axis_angle(point) for point in x])
                    x = unnormalize_data(x_axis_angle)

                    c_axis_angle = torch.stack([convert_to_axis_angle(point) for point in c])
                    c = unnormalize_data(c_axis_angle)

                    x_syn_full_axis_angle = torch.stack([convert_to_axis_angle(point) for point in x_syn_full])
                    x_syn = unnormalize_data(x_syn_full_axis_angle)
                    
                    print(f'plotting validation for epoch {i}')
                    vis_results(c, x, x_syn, i, logging['vis_test_results_path'], is_smplx=True)
                    break
                
                

        
        

        
    return losses_G_transl, losses_MSE_transl, losses_V_MSE_transl, losses_G_poses, losses_MSE_poses, losses_V_MSE_poses, losses_MSE_full, losses_V_MSE_full