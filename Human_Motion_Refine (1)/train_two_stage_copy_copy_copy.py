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

writer = SummaryWriter("/scratch/gilbreth/anugu/logs_demo/demo_6_models_one_point_along_with_adv_hm_bl_loss_v1")

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
    print('the prob reaches here')
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
    print('the prob reaches here too ')
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

def seperate_data(data):
    '''data is (N, T, 242)'''
    move = data[:, :, :3]
    orient = data[:, :, 3:6]
    betas = data[:, :, 226:242]
    
    poses = data[:, :, 6:226]

    hands = torch.cat([poses[:, :, 4*13:4*15], poses[:, :, 4*16:4*22], poses[:, :, 4*25:4*55]], axis=2)
    legs = torch.cat([poses[:, :, 4*1:4*3], poses[:, :, 4*4:4*6], poses[:, :, 4*7:4*9], poses[:, :, 4*10:4*12]], axis=2)
    torso = torch.cat([poses[:, :, 4*0:4*1], poses[:, :, 4*3:4*4], poses[:, :, 4*6:4*7], poses[:, :, 4*9:4*10], poses[:, :, 4*12:4*13], poses[:, :, 4*15:4*16], poses[:, :, 4*22:4*25]], axis=2)

    return move, orient, betas, hands, legs, torso

def combine_data(move, orient, betas, hands, legs, torso):
    '''Recombine the separated components into the original data format (N, T, 242)'''

    # Initialize an empty tensor to hold the poses data
    N, T, _ = move.shape
    poses = torch.zeros((N, T, 220), dtype=move.dtype, device=move.device)

    # Combine hands back into poses
    poses[:, :, 4*13:4*15] = hands[:, :, :8]
    poses[:, :, 4*16:4*22] = hands[:, :, 8:32]
    poses[:, :, 4*25:4*55] = hands[:, :, 32:]

    # Combine legs back into poses
    poses[:, :, 4*1:4*3] = legs[:, :, :8]
    poses[:, :, 4*4:4*6] = legs[:, :, 8:16]
    poses[:, :, 4*7:4*9] = legs[:, :, 16:24]
    poses[:, :, 4*10:4*12] = legs[:, :, 24:]

    # Combine torso back into poses
    poses[:, :, 4*0:4*1] = torso[:, :, :4]
    poses[:, :, 4*3:4*4] = torso[:, :, 4:8]
    poses[:, :, 4*6:4*7] = torso[:, :, 8:12]
    poses[:, :, 4*9:4*10] = torso[:, :, 12:16]
    poses[:, :, 4*12:4*13] = torso[:, :, 16:20]
    poses[:, :, 4*15:4*16] = torso[:, :, 20:24]
    poses[:, :, 4*22:4*25] = torso[:, :, 24:]

    # Concatenate all components to form the original data format
    data = torch.cat([move, orient, poses, betas], axis=2)

    return data

def train_discriminator(x, c, D, G, D_optimizer, D_scheduler, λ_grad, grf, device):
    res = x.shape[1]
    D_optimizer.zero_grad()

    x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)

    W_loss = -torch.mean(D(x, c)) + torch.mean(D(x_syn.detach(), c))

    gp = calculate_gradient_penalty(D, x.data, c, x_syn.data, c, device, res)

    loss_D = W_loss + λ_grad * gp
    loss_D.backward()

    D_optimizer.step()
    D_scheduler.step()

    return x_syn, loss_D.item(), W_loss.item()

def train_generator(x, c, G, G_optimizer, G_scheduler, gt_weight, grf):
    G_optimizer.zero_grad()

    x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)

    loss_G = gt_weight*F.mse_loss(x_syn, x)
    loss_G.backward()

    G_optimizer.step()
    G_scheduler.step()

    return x_syn, loss_G.item()

def train_generator_actual(x, c, D, G, G_optimizer, G_scheduler, gt_weight, grf):
    G_optimizer.zero_grad()

    x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)

    loss_G = gt_weight*F.mse_loss(x_syn, x) - torch.mean(D(x_syn, c))
    loss_G.backward(retain_graph=True)

    G_optimizer.step()
    G_scheduler.step()

    return x_syn, loss_G.item()


    


def train_GANO_HM_two_stage_copy_copy_copy(D_move, D_orient, D_betas, D_hands, D_legs, D_torso,
                                           G_move, G_orient, G_betas, G_hands, G_legs, G_torso,
                                            train_dataloader, val_dataloader, epochs, D_move_optimizer, D_orient_optimizer, D_betas_optimizer, D_hands_optimizer, D_legs_optimizer, D_torso_optimizer,
                                            G_move_optimizer, G_orient_optimizer, G_betas_optimizer, G_hands_optimizer, G_legs_optimizer, G_torso_optimizer,
                                            λ_grad, grf, n_critic, device, batch_size, logging, hm_weight, fc_weight, gt_weight,
                                            D_move_scheduler, D_orient_scheduler, D_betas_scheduler, D_hands_scheduler, D_legs_scheduler, D_torso_scheduler,
                                            G_move_scheduler, G_orient_scheduler, G_betas_scheduler, G_hands_scheduler, G_legs_scheduler, G_torso_scheduler, 
                                            data_stats = None, scheduler=None):

    losses_D_move = np.zeros(epochs)
    losses_G_move = np.zeros(epochs)
    losses_W_move = np.zeros(epochs)
    losses_MSE_move = np.zeros(epochs)
    losses_V_MSE_move = np.zeros(epochs)
    
    losses_D_orient = np.zeros(epochs)
    losses_G_orient = np.zeros(epochs)
    losses_W_orient = np.zeros(epochs)
    losses_MSE_orient = np.zeros(epochs)
    losses_V_MSE_orient = np.zeros(epochs)

    losses_D_betas = np.zeros(epochs)
    losses_G_betas = np.zeros(epochs)
    losses_W_betas = np.zeros(epochs)
    losses_MSE_betas = np.zeros(epochs)
    losses_V_MSE_betas = np.zeros(epochs)

    losses_D_hands = np.zeros(epochs)
    losses_G_hands = np.zeros(epochs)
    losses_W_hands = np.zeros(epochs)
    losses_MSE_hands = np.zeros(epochs)
    losses_V_MSE_hands = np.zeros(epochs)

    losses_D_legs = np.zeros(epochs)
    losses_G_legs = np.zeros(epochs)
    losses_W_legs = np.zeros(epochs)
    losses_MSE_legs = np.zeros(epochs)
    losses_V_MSE_legs = np.zeros(epochs)

    losses_D_torso = np.zeros(epochs)
    losses_G_torso = np.zeros(epochs)
    losses_W_torso = np.zeros(epochs)
    losses_MSE_torso = np.zeros(epochs)
    losses_V_MSE_torso = np.zeros(epochs)

    no_of_batches = len(train_dataloader)

    losses_MSE_full = np.zeros(epochs)
    losses_V_MSE_full = np.zeros(epochs)

    for i in tqdm(range(epochs)):
        loss_D_move = 0.0
        loss_G_move = 0.0
        loss_W_move = 0.0
        loss_MSE_move = 0.0
        loss_V_MSE_move = 0.0

        loss_D_orient = 0.0
        loss_G_orient = 0.0
        loss_W_orient = 0.0
        loss_MSE_orient = 0.0
        loss_V_MSE_orient = 0.0

        loss_D_betas = 0.0
        loss_G_betas = 0.0
        loss_W_betas = 0.0
        loss_MSE_betas = 0.0
        loss_V_MSE_betas = 0.0

        loss_D_hands = 0.0
        loss_G_hands = 0.0
        loss_W_hands = 0.0
        loss_MSE_hands = 0.0
        loss_V_MSE_hands = 0.0

        loss_D_legs = 0.0
        loss_G_legs = 0.0
        loss_W_legs = 0.0
        loss_MSE_legs = 0.0
        loss_V_MSE_legs = 0.0

        loss_D_torso = 0.0
        loss_G_torso = 0.0
        loss_W_torso = 0.0
        loss_MSE_torso = 0.0
        loss_V_MSE_torso = 0.0

        loss_MSE_full = 0.0
        loss_V_MSE_full = 0.0

        for j, data in tqdm(enumerate(train_dataloader)):

            if random.random()<0.5:
                x = data[0].to(device).float()
                c = data[1].to(device).float()

                x_move, x_orient, x_betas, x_hands, x_legs, x_torso = seperate_data(x)
                c_move, c_orient, c_betas, c_hands, c_legs, c_torso = seperate_data(c)

            else:
                x = data[1].to(device).float()
                c = data[0].to(device).float()

                x_move, x_orient, x_betas, x_hands, x_legs, x_torso = seperate_data(x)
                c_move, c_orient, c_betas, c_hands, c_legs, c_torso = seperate_data(c)
            
            #-------- Training D_move --------#

            x_move = x_move.reshape(x_move.shape[0], x_move.shape[1], -1)
            c_move = c_move.reshape(c_move.shape[0], c_move.shape[1], -1)

            x_syn_move, loss_D, loss_W = train_discriminator(x_move, c_move, D_move, G_move, D_move_optimizer, D_move_scheduler, λ_grad, grf, device)
            loss_D_move += loss_D
            loss_W_move += loss_W

            #-------- Training D_orient --------#

            x_orient = x_orient.reshape(x_orient.shape[0], x_orient.shape[1], -1)
            c_orient = c_orient.reshape(c_orient.shape[0], c_orient.shape[1], -1)

            x_syn_orient, loss_D, loss_W = train_discriminator(x_orient, c_orient, D_orient, G_orient, D_orient_optimizer, D_orient_scheduler, λ_grad, grf, device)
            loss_D_orient += loss_D
            loss_W_orient += loss_W

            #-------- Training D_betas --------#

            x_betas = x_betas.reshape(x_betas.shape[0], x_betas.shape[1], -1)
            c_betas = c_betas.reshape(c_betas.shape[0], c_betas.shape[1], -1)

            x_syn_betas, loss_D, loss_W = train_discriminator(x_betas, c_betas, D_betas, G_betas, D_betas_optimizer, D_betas_scheduler, λ_grad, grf, device)
            loss_D_betas += loss_D
            loss_W_betas += loss_W

            #-------- Training D_hands --------#

            x_hands = x_hands.reshape(x_hands.shape[0], x_hands.shape[1], -1)
            c_hands = c_hands.reshape(c_hands.shape[0], c_hands.shape[1], -1)

            x_syn_hands, loss_D, loss_W = train_discriminator(x_hands, c_hands, D_hands, G_hands, D_hands_optimizer, D_hands_scheduler, λ_grad, grf, device)
            loss_D_hands += loss_D
            loss_W_hands += loss_W

            #-------- Training D_legs --------#

            x_legs = x_legs.reshape(x_legs.shape[0], x_legs.shape[1], -1)
            c_legs = c_legs.reshape(c_legs.shape[0], c_legs.shape[1], -1)

            x_syn_legs, loss_D, loss_W = train_discriminator(x_legs, c_legs, D_legs, G_legs, D_legs_optimizer, D_legs_scheduler, λ_grad, grf, device)
            loss_D_legs += loss_D
            loss_W_legs += loss_W

            #-------- Training D_torso --------#

            x_torso = x_torso.reshape(x_torso.shape[0], x_torso.shape[1], -1)
            c_torso = c_torso.reshape(c_torso.shape[0], c_torso.shape[1], -1)

            x_syn_torso, loss_D, loss_W = train_discriminator(x_torso, c_torso, D_torso, G_torso, D_torso_optimizer, D_torso_scheduler, λ_grad, grf, device)
            loss_D_torso += loss_D
            loss_W_torso += loss_W

            x_syn_full = combine_data(x_syn_move, x_syn_orient, x_syn_betas, x_syn_hands, x_syn_legs, x_syn_torso)
            
            
            #-------- Train Generators --------#
            if (j + 1) % n_critic == 0:

                #-------- Training G_move --------#

                x_syn_move, loss_G = train_generator_actual(x_move, c_move, D_move, G_move, G_move_optimizer, G_move_scheduler, gt_weight, grf)
                loss_G_move += loss_G

                #-------- Training G_orient --------#

                x_syn_orient, loss_G = train_generator_actual(x_orient, c_orient, D_orient, G_orient, G_orient_optimizer, G_orient_scheduler, gt_weight, grf)
                loss_G_orient += loss_G

                #-------- Training G_betas --------#

                x_syn_betas, loss_G = train_generator_actual(x_betas, c_betas, D_betas, G_betas, G_betas_optimizer, G_betas_scheduler, gt_weight, grf)
                loss_G_betas += loss_G

                #-------- Training G_hands --------#

                x_syn_hands, loss_G = train_generator_actual(x_hands, c_hands, D_hands, G_hands, G_hands_optimizer, G_hands_scheduler, gt_weight, grf)
                loss_G_hands += loss_G

                #-------- Training G_legs --------#

                x_syn_legs, loss_G = train_generator_actual(x_legs, c_legs, D_legs, G_legs, G_legs_optimizer, G_legs_scheduler, gt_weight, grf)
                loss_G_legs += loss_G

                #-------- Training G_torso --------#

                x_syn_torso, loss_G = train_generator_actual(x_torso, c_torso, D_torso, G_torso, G_torso_optimizer, G_torso_scheduler, gt_weight, grf)
                loss_G_torso += loss_G

                #-------- Training G_full --------#

                x_syn_full = combine_data(x_syn_move, x_syn_orient, x_syn_betas, x_syn_hands, x_syn_legs, x_syn_torso)


            loss_MSE_move += F.mse_loss(x_syn_move, x_move)
            loss_MSE_orient += F.mse_loss(x_syn_orient, x_orient)
            loss_MSE_betas += F.mse_loss(x_syn_betas, x_betas)
            loss_MSE_hands += F.mse_loss(x_syn_hands, x_hands)
            loss_MSE_legs += F.mse_loss(x_syn_legs, x_legs)
            loss_MSE_torso += F.mse_loss(x_syn_torso, x_torso)
            loss_MSE_full += (F.mse_loss(x_syn_move, x_move) + F.mse_loss(x_syn_orient, x_orient) + F.mse_loss(x_syn_betas, x_betas) + F.mse_loss(x_syn_hands, x_hands) + F.mse_loss(x_syn_legs, x_legs) + F.mse_loss(x_syn_torso, x_torso))/6

            del x, c, x_move, c_move, x_orient, c_orient, x_betas, c_betas, x_hands, c_hands, x_legs, c_legs, x_torso, c_torso
            del x_syn_move, x_syn_orient, x_syn_betas, x_syn_hands, x_syn_legs, x_syn_torso, x_syn_full
            gc.collect()
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            no_of_val_batches = len(val_dataloader)
            for j, data in enumerate(val_dataloader):
                x = data[0].to(device).float()
                c = data[1].to(device).float()

                x_move, x_orient, x_betas, x_hands, x_legs, x_torso = seperate_data(x)
                c_move, c_orient, c_betas, c_hands, c_legs, c_torso = seperate_data(c)

                x_move = x_move.reshape(x_move.shape[0], x_move.shape[1], -1)
                c_move = c_move.reshape(c_move.shape[0], c_move.shape[1], -1)

                x_orient = x_orient.reshape(x_orient.shape[0], x_orient.shape[1], -1)
                c_orient = c_orient.reshape(c_orient.shape[0], c_orient.shape[1], -1)

                x_betas = x_betas.reshape(x_betas.shape[0], x_betas.shape[1], -1)
                c_betas = c_betas.reshape(c_betas.shape[0], c_betas.shape[1], -1)

                x_hands = x_hands.reshape(x_hands.shape[0], x_hands.shape[1], -1)
                c_hands = c_hands.reshape(c_hands.shape[0], c_hands.shape[1], -1)

                x_legs = x_legs.reshape(x_legs.shape[0], x_legs.shape[1], -1)
                c_legs = c_legs.reshape(c_legs.shape[0], c_legs.shape[1], -1)

                x_torso = x_torso.reshape(x_torso.shape[0], x_torso.shape[1], -1)
                c_torso = c_torso.reshape(c_torso.shape[0], c_torso.shape[1], -1)

                x_syn_move = G_move(grf.sample(x_move.shape[0], x_move.shape[1]).float(),c_move)
                x_syn_orient = G_orient(grf.sample(x_orient.shape[0], x_orient.shape[1]).float(),c_orient)
                x_syn_betas = G_betas(grf.sample(x_betas.shape[0], x_betas.shape[1]).float(),c_betas)
                x_syn_hands = G_hands(grf.sample(x_hands.shape[0], x_hands.shape[1]).float(),c_hands)
                x_syn_legs = G_legs(grf.sample(x_legs.shape[0], x_legs.shape[1]).float(),c_legs)
                x_syn_torso = G_torso(grf.sample(x_torso.shape[0], x_torso.shape[1]).float(),c_torso)
                x_syn_full = combine_data(x_syn_move, x_syn_orient, x_syn_betas, x_syn_hands, x_syn_legs, x_syn_torso)

                loss_V_MSE_move += F.mse_loss(x_syn_move, x_move)
                loss_V_MSE_orient += F.mse_loss(x_syn_orient, x_orient)
                loss_V_MSE_betas += F.mse_loss(x_syn_betas, x_betas)
                loss_V_MSE_hands += F.mse_loss(x_syn_hands, x_hands)
                loss_V_MSE_legs += F.mse_loss(x_syn_legs, x_legs)
                loss_V_MSE_torso += F.mse_loss(x_syn_torso, x_torso)
                loss_V_MSE_full += (F.mse_loss(x_syn_move, x_move) + F.mse_loss(x_syn_orient, x_orient) + F.mse_loss(x_syn_betas, x_betas) + F.mse_loss(x_syn_hands, x_hands) + F.mse_loss(x_syn_legs, x_legs) + F.mse_loss(x_syn_torso, x_torso))/6


                del x, c, x_move, c_move, x_orient, c_orient, x_betas, c_betas, x_hands, c_hands, x_legs, c_legs, x_torso, c_torso
                del x_syn_move, x_syn_orient, x_syn_betas, x_syn_hands, x_syn_legs, x_syn_torso, x_syn_full
                gc.collect()
                torch.cuda.empty_cache()
        
        losses_G_move[i] = loss_G_move / (no_of_batches/n_critic)
        losses_W_move[i] = loss_W_move / no_of_batches
        losses_D_move[i] = loss_D_move / no_of_batches
        losses_MSE_move[i] = loss_MSE_move/ no_of_batches
        losses_V_MSE_move[i] =  loss_V_MSE_move/ no_of_val_batches

        losses_G_orient[i] = loss_G_orient / (no_of_batches/n_critic)
        losses_W_orient[i] = loss_W_orient / no_of_batches
        losses_D_orient[i] = loss_D_orient / no_of_batches
        losses_MSE_orient[i] = loss_MSE_orient/ no_of_batches
        losses_V_MSE_orient[i] =  loss_V_MSE_orient/ no_of_val_batches

        losses_G_betas[i] = loss_G_betas / (no_of_batches/n_critic)
        losses_W_betas[i] = loss_W_betas / no_of_batches
        losses_D_betas[i] = loss_D_betas / no_of_batches
        losses_MSE_betas[i] = loss_MSE_betas/ no_of_batches
        losses_V_MSE_betas[i] =  loss_V_MSE_betas/ no_of_val_batches

        losses_G_hands[i] = loss_G_hands / (no_of_batches/n_critic)
        losses_W_hands[i] = loss_W_hands / no_of_batches
        losses_D_hands[i] = loss_D_hands / no_of_batches
        losses_MSE_hands[i] = loss_MSE_hands/ no_of_batches
        losses_V_MSE_hands[i] =  loss_V_MSE_hands/ no_of_val_batches

        losses_G_legs[i] = loss_G_legs / (no_of_batches/n_critic)
        losses_W_legs[i] = loss_W_legs / no_of_batches
        losses_D_legs[i] = loss_D_legs / no_of_batches
        losses_MSE_legs[i] = loss_MSE_legs/ no_of_batches
        losses_V_MSE_legs[i] =  loss_V_MSE_legs/ no_of_val_batches

        losses_G_torso[i] = loss_G_torso / (no_of_batches/n_critic)
        losses_W_torso[i] = loss_W_torso / no_of_batches
        losses_D_torso[i] = loss_D_torso / no_of_batches
        losses_MSE_torso[i] = loss_MSE_torso/ no_of_batches
        losses_V_MSE_torso[i] =  loss_V_MSE_torso/ no_of_val_batches

        losses_MSE_full[i] = loss_MSE_full/ no_of_batches
        losses_V_MSE_full[i] = loss_V_MSE_full/ no_of_val_batches

        # Add to tensorboard.
        writer.add_scalar('losses_G_move', losses_G_move[i], i)
        writer.add_scalar('losses_W_move', losses_W_move[i], i)
        writer.add_scalar('losses_D_move', losses_D_move[i], i)
        writer.add_scalar('losses_MSE_move', losses_MSE_move[i], i)
        writer.add_scalar('losses_V_MSE_move', losses_V_MSE_move[i], i)

        writer.add_scalar('losses_G_orient', losses_G_orient[i], i)
        writer.add_scalar('losses_W_orient', losses_W_orient[i], i)
        writer.add_scalar('losses_D_orient', losses_D_orient[i], i)
        writer.add_scalar('losses_MSE_orient', losses_MSE_orient[i], i)
        writer.add_scalar('losses_V_MSE_orient', losses_V_MSE_orient[i], i)

        writer.add_scalar('losses_G_betas', losses_G_betas[i], i)
        writer.add_scalar('losses_W_betas', losses_W_betas[i], i)
        writer.add_scalar('losses_D_betas', losses_D_betas[i], i)
        writer.add_scalar('losses_MSE_betas', losses_MSE_betas[i], i)
        writer.add_scalar('losses_V_MSE_betas', losses_V_MSE_betas[i], i)

        writer.add_scalar('losses_G_hands', losses_G_hands[i], i)
        writer.add_scalar('losses_W_hands', losses_W_hands[i], i)
        writer.add_scalar('losses_D_hands', losses_D_hands[i], i)
        writer.add_scalar('losses_MSE_hands', losses_MSE_hands[i], i)
        writer.add_scalar('losses_V_MSE_hands', losses_V_MSE_hands[i], i)

        writer.add_scalar('losses_G_legs', losses_G_legs[i], i)
        writer.add_scalar('losses_W_legs', losses_W_legs[i], i)
        writer.add_scalar('losses_D_legs', losses_D_legs[i], i)
        writer.add_scalar('losses_MSE_legs', losses_MSE_legs[i], i)
        writer.add_scalar('losses_V_MSE_legs', losses_V_MSE_legs[i], i)

        writer.add_scalar('losses_G_torso', losses_G_torso[i], i)
        writer.add_scalar('losses_W_torso', losses_W_torso[i], i)
        writer.add_scalar('losses_D_torso', losses_D_torso[i], i)
        writer.add_scalar('losses_MSE_torso', losses_MSE_torso[i], i)
        writer.add_scalar('losses_V_MSE_torso', losses_V_MSE_torso[i], i)

        writer.add_scalar('loss_MSE_full', losses_MSE_full[i], i)
        writer.add_scalar('loss_V_MSE_full', losses_V_MSE_full[i], i)

        if i%100==0:
            torch.save(G_move.state_dict(), logging['G_move_save_path']+f"generator_move_epoch{i}.pt")
            torch.save(G_orient.state_dict(), logging['G_orient_save_path']+f"generator_orient_epoch{i}.pt")
            torch.save(G_betas.state_dict(), logging['G_betas_save_path']+f"generator_betas_epoch{i}.pt")
            torch.save(G_hands.state_dict(), logging['G_hands_save_path']+f"generator_hands_epoch{i}.pt")
            torch.save(G_legs.state_dict(), logging['G_legs_save_path']+f"generator_legs_epoch{i}.pt")
            torch.save(G_torso.state_dict(), logging['G_torso_save_path']+f"generator_torso_epoch{i}.pt")

            torch.save(D_move.state_dict(), logging['D_move_save_path']+f"discriminator_move_epoch{i}.pt")
            torch.save(D_orient.state_dict(), logging['D_orient_save_path']+f"discriminator_orient_epoch{i}.pt")
            torch.save(D_betas.state_dict(), logging['D_betas_save_path']+f"discriminator_betas_epoch{i}.pt")
            torch.save(D_hands.state_dict(), logging['D_hands_save_path']+f"discriminator_hands_epoch{i}.pt")
            torch.save(D_legs.state_dict(), logging['D_legs_save_path']+f"discriminator_legs_epoch{i}.pt")
            torch.save(D_torso.state_dict(), logging['D_torso_save_path']+f"discriminator_torso_epoch{i}.pt")

            torch.save(G_move_optimizer.state_dict(), logging['G_move_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(G_orient_optimizer.state_dict(), logging['G_orient_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(G_betas_optimizer.state_dict(), logging['G_betas_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(G_hands_optimizer.state_dict(), logging['G_hands_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(G_legs_optimizer.state_dict(), logging['G_legs_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(G_torso_optimizer.state_dict(), logging['G_torso_optimizer_save_path']+f"epoch{i}.pt")

            torch.save(D_move_optimizer.state_dict(), logging['D_move_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(D_orient_optimizer.state_dict(), logging['D_orient_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(D_betas_optimizer.state_dict(), logging['D_betas_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(D_hands_optimizer.state_dict(), logging['D_hands_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(D_legs_optimizer.state_dict(), logging['D_legs_optimizer_save_path']+f"epoch{i}.pt")
            torch.save(D_torso_optimizer.state_dict(), logging['D_torso_optimizer_save_path']+f"epoch{i}.pt")


            torch.save(torch.from_numpy(losses_G_move), logging['plots']+f"loss_G_move.pt")
            torch.save(torch.from_numpy(losses_W_move), logging['plots']+f"loss_W_move.pt")
            torch.save(torch.from_numpy(losses_D_move), logging['plots']+f"loss_D_move.pt")
            torch.save(torch.from_numpy(losses_MSE_move), logging['plots']+f"loss_MSE_move.pt")
            torch.save(torch.from_numpy(losses_V_MSE_move), logging['plots']+f"loss_V_MSE_move.pt")

            torch.save(torch.from_numpy(losses_G_orient), logging['plots']+f"loss_G_orient.pt")
            torch.save(torch.from_numpy(losses_W_orient), logging['plots']+f"loss_W_orient.pt")
            torch.save(torch.from_numpy(losses_D_orient), logging['plots']+f"loss_D_orient.pt")
            torch.save(torch.from_numpy(losses_MSE_orient), logging['plots']+f"loss_MSE_orient.pt")
            torch.save(torch.from_numpy(losses_V_MSE_orient), logging['plots']+f"loss_V_MSE_orient.pt")

            torch.save(torch.from_numpy(losses_G_betas), logging['plots']+f"loss_G_betas.pt")
            torch.save(torch.from_numpy(losses_W_betas), logging['plots']+f"loss_W_betas.pt")
            torch.save(torch.from_numpy(losses_D_betas), logging['plots']+f"loss_D_betas.pt")
            torch.save(torch.from_numpy(losses_MSE_betas), logging['plots']+f"loss_MSE_betas.pt")
            torch.save(torch.from_numpy(losses_V_MSE_betas), logging['plots']+f"loss_V_MSE_betas.pt")

            torch.save(torch.from_numpy(losses_G_hands), logging['plots']+f"loss_G_hands.pt")
            torch.save(torch.from_numpy(losses_W_hands), logging['plots']+f"loss_W_hands.pt")
            torch.save(torch.from_numpy(losses_D_hands), logging['plots']+f"loss_D_hands.pt")
            torch.save(torch.from_numpy(losses_MSE_hands), logging['plots']+f"loss_MSE_hands.pt")
            torch.save(torch.from_numpy(losses_V_MSE_hands), logging['plots']+f"loss_V_MSE_hands.pt")

            torch.save(torch.from_numpy(losses_G_legs), logging['plots']+f"loss_G_legs.pt")
            torch.save(torch.from_numpy(losses_W_legs), logging['plots']+f"loss_W_legs.pt")
            torch.save(torch.from_numpy(losses_D_legs), logging['plots']+f"loss_D_legs.pt")
            torch.save(torch.from_numpy(losses_MSE_legs), logging['plots']+f"loss_MSE_legs.pt")
            torch.save(torch.from_numpy(losses_V_MSE_legs), logging['plots']+f"loss_V_MSE_legs.pt")

            torch.save(torch.from_numpy(losses_G_torso), logging['plots']+f"loss_G_torso.pt")
            torch.save(torch.from_numpy(losses_W_torso), logging['plots']+f"loss_W_torso.pt")
            torch.save(torch.from_numpy(losses_D_torso), logging['plots']+f"loss_D_torso.pt")
            torch.save(torch.from_numpy(losses_MSE_torso), logging['plots']+f"loss_MSE_torso.pt")
            torch.save(torch.from_numpy(losses_V_MSE_torso), logging['plots']+f"loss_V_MSE_torso.pt")

            torch.save(torch.from_numpy(losses_MSE_full), logging['plots']+f"loss_MSE_full.pt")
            torch.save(torch.from_numpy(losses_V_MSE_full), logging['plots']+f"loss_V_MSE_full.pt")


        if i%100==0:
            with torch.no_grad():
                for j, data in enumerate(val_dataloader):

                    x = data[0].to(device).float()
                    c = data[1].to(device).float()

                    x_move, x_orient, x_betas, x_hands, x_legs, x_torso = seperate_data(x)
                    c_move, c_orient, c_betas, c_hands, c_legs, c_torso = seperate_data(c)

                    x_move = x_move.reshape(x_move.shape[0], x_move.shape[1], -1)
                    c_move = c_move.reshape(c_move.shape[0], c_move.shape[1], -1)

                    x_orient = x_orient.reshape(x_orient.shape[0], x_orient.shape[1], -1)
                    c_orient = c_orient.reshape(c_orient.shape[0], c_orient.shape[1], -1)

                    x_betas = x_betas.reshape(x_betas.shape[0], x_betas.shape[1], -1)
                    c_betas = c_betas.reshape(c_betas.shape[0], c_betas.shape[1], -1)

                    x_hands = x_hands.reshape(x_hands.shape[0], x_hands.shape[1], -1)
                    c_hands = c_hands.reshape(c_hands.shape[0], c_hands.shape[1], -1)

                    x_legs = x_legs.reshape(x_legs.shape[0], x_legs.shape[1], -1)
                    c_legs = c_legs.reshape(c_legs.shape[0], c_legs.shape[1], -1)

                    x_torso = x_torso.reshape(x_torso.shape[0], x_torso.shape[1], -1)
                    c_torso = c_torso.reshape(c_torso.shape[0], c_torso.shape[1], -1)

                    x_syn_move = G_move(grf.sample(x_move.shape[0], x_move.shape[1]).float(),c_move)
                    x_syn_orient = G_orient(grf.sample(x_orient.shape[0], x_orient.shape[1]).float(),c_orient)
                    x_syn_betas = G_betas(grf.sample(x_betas.shape[0], x_betas.shape[1]).float(),c_betas)
                    x_syn_hands = G_hands(grf.sample(x_hands.shape[0], x_hands.shape[1]).float(),c_hands)
                    x_syn_legs = G_legs(grf.sample(x_legs.shape[0], x_legs.shape[1]).float(),c_legs)
                    x_syn_torso = G_torso(grf.sample(x_torso.shape[0], x_torso.shape[1]).float(),c_torso)
                    x_syn_full = combine_data(x_syn_move, x_syn_orient, x_syn_betas, x_syn_hands, x_syn_legs, x_syn_torso)

                    x_axis_angle = torch.stack([convert_to_axis_angle(point) for point in x])
                    x = unnormalize_data(x_axis_angle)

                    c_axis_angle = torch.stack([convert_to_axis_angle(point) for point in c])
                    c = unnormalize_data(c_axis_angle)

                    x_syn_full_axis_angle = torch.stack([convert_to_axis_angle(point) for point in x_syn_full])
                    x_syn = unnormalize_data(x_syn_full_axis_angle)

                    print(f'plotting validation for epoch {i}')
                    vis_results(c, x, x_syn, i, logging['vis_test_results_path'], is_smplx=True)
                    break

                '''for j, data in enumerate(train_dataloader):

                    x = data[0].to(device).float()
                    c = data[1].to(device).float()

                    x_move, x_orient, x_betas, x_hands, x_legs, x_torso = seperate_data(x)
                    c_move, c_orient, c_betas, c_hands, c_legs, c_torso = seperate_data(c)

                    x_move = x_move.reshape(x_move.shape[0], x_move.shape[1], -1)
                    c_move = c_move.reshape(c_move.shape[0], c_move.shape[1], -1)

                    x_orient = x_orient.reshape(x_orient.shape[0], x_orient.shape[1], -1)
                    c_orient = c_orient.reshape(c_orient.shape[0], c_orient.shape[1], -1)

                    x_betas = x_betas.reshape(x_betas.shape[0], x_betas.shape[1], -1)
                    c_betas = c_betas.reshape(c_betas.shape[0], c_betas.shape[1], -1)

                    x_hands = x_hands.reshape(x_hands.shape[0], x_hands.shape[1], -1)
                    c_hands = c_hands.reshape(c_hands.shape[0], c_hands.shape[1], -1)

                    x_legs = x_legs.reshape(x_legs.shape[0], x_legs.shape[1], -1)
                    c_legs = c_legs.reshape(c_legs.shape[0], c_legs.shape[1], -1)

                    x_torso = x_torso.reshape(x_torso.shape[0], x_torso.shape[1], -1)
                    c_torso = c_torso.reshape(c_torso.shape[0], c_torso.shape[1], -1)

                    x_syn_move = G_move(grf.sample(x_move.shape[0], x_move.shape[1]).float(),c_move)
                    x_syn_orient = G_orient(grf.sample(x_orient.shape[0], x_orient.shape[1]).float(),c_orient)
                    x_syn_betas = G_betas(grf.sample(x_betas.shape[0], x_betas.shape[1]).float(),c_betas)
                    x_syn_hands = G_hands(grf.sample(x_hands.shape[0], x_hands.shape[1]).float(),c_hands)
                    x_syn_legs = G_legs(grf.sample(x_legs.shape[0], x_legs.shape[1]).float(),c_legs)
                    x_syn_torso = G_torso(grf.sample(x_torso.shape[0], x_torso.shape[1]).float(),c_torso)
                    x_syn_full = combine_data(x_syn_move, x_syn_orient, x_syn_betas, x_syn_hands, x_syn_legs, x_syn_torso)

                    x_axis_angle = torch.stack([convert_to_axis_angle(point) for point in x])
                    x = unnormalize_data(x_axis_angle)

                    c_axis_angle = torch.stack([convert_to_axis_angle(point) for point in c])
                    c = unnormalize_data(c_axis_angle)

                    x_syn_full_axis_angle = torch.stack([convert_to_axis_angle(point) for point in x_syn_full])
                    x_syn = unnormalize_data(x_syn_full_axis_angle)

                    print(f'plotting training for epoch {i}')
                    vis_results(c, x, x_syn, i, logging['vis_train_results_path'], is_smplx=True)
                    break'''

    final_result = {'losses_G_move': losses_G_move, 'losses_MSE_move': losses_MSE_move, 'losses_V_MSE_move': losses_V_MSE_move,
                    'losses_G_orient': losses_G_orient, 'losses_MSE_orient': losses_MSE_orient, 'losses_V_MSE_orient': losses_V_MSE_orient,
                    'losses_G_betas': losses_G_betas, 'losses_MSE_betas': losses_MSE_betas, 'losses_V_MSE_betas': losses_V_MSE_betas, 
                    'losses_G_hands': losses_G_hands, 'losses_MSE_hands': losses_MSE_hands, 'losses_V_MSE_hands': losses_V_MSE_hands, 
                    'losses_G_legs': losses_G_legs, 'losses_MSE_legs': losses_MSE_legs, 'losses_V_MSE_legs': losses_V_MSE_legs, 
                    'losses_G_torso': losses_G_torso, 'losses_MSE_torso': losses_MSE_torso, 'losses_V_MSE_torso': losses_V_MSE_torso, 
                    'losses_MSE_full': losses_MSE_full, 'losses_V_MSE_full': losses_V_MSE_full}

    return final_result

        