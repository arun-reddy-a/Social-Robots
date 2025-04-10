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

writer = SummaryWriter("/scratch/gilbreth/anugu/logs_demo/demo_2")

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

def human_loss(data,lamda = 0.0001):
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
    return lamda*error


def train_GANO_HM(D, G, train_data, val_data, epochs, D_optim, G_optim, λ_grad, grf, n_critic, device, batch_size, logging, data_stats = None, scheduler=None):
    losses_D = np.zeros(epochs)
    losses_G = np.zeros(epochs)
    losses_W = np.zeros(epochs)
    losses_MSE = np.zeros(epochs)
    losses_V_MSE = np.zeros(epochs)
    no_of_batches = len(train_data)

    for i in tqdm(range(epochs)):
        loss_D = 0.0
        loss_G = 0.0
        loss_W = 0.0
        loss_MSE = 0.0
        loss_V_MSE = 0.0
        for j, data in tqdm(enumerate(train_data)):

            if random.random()<0.5:
                x = data[0].to(device).float()
                c = data[1].to(device).float()
            else:
                x = data[1].to(device).float()
                c = data[0].to(device).float()

            x = x.reshape(x.shape[0], x.shape[1], -1)
            c = c.reshape(c.shape[0], c.shape[1], -1)

            res = x.shape[1] #asuming uniform resolution along x and y
            D_optim.zero_grad()
            
            x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)

            W_loss = -torch.mean(D(x,c)) + torch.mean(D(x_syn.detach(),c))

            gradient_penalty = calculate_gradient_penalty(D, x.data,c, x_syn.data,c, device, res)

            loss = W_loss + λ_grad * gradient_penalty
            loss.backward()

            loss_D += loss.item()
            loss_W += W_loss.item()

            D_optim.step()

            del res, gradient_penalty
            gc.collect()
            torch.cuda.empty_cache()
            
            # Train G
            if (j + 1) % n_critic == 0:
                G_optim.zero_grad()

                x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)

                # loss = -torch.mean(D(x_syn,c))+human_loss(x_syn.reshape(-1,res,55,3))
                loss = -torch.mean(D(x_syn,c))
                loss.backward()
                loss_G += loss.item()

                G_optim.step()

                gc.collect()
                torch.cuda.empty_cache()
            
            loss_MSE += F.mse_loss(x_syn, x)  
        
        with torch.no_grad():
            no_of_val_batches = len(val_data)
            for j, data in enumerate(val_data):
                x = data[0].to(device).float()
                c = data[1].to(device).float()
                x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)
                loss_V_MSE += F.mse_loss(x_syn, x)



        losses_MSE[i] = loss_MSE/no_of_batches
        losses_V_MSE[i] =  loss_V_MSE/no_of_val_batches  
        losses_D[i] = loss_D / no_of_batches
        losses_G[i] = loss_G / (no_of_batches/n_critic)
        losses_W[i] = loss_W / no_of_batches

        # Add to tensorboard.
        writer.add_scalar('losses_MSE', losses_MSE[i], i)
        writer.add_scalar('losses_V_MSE', losses_V_MSE[i], i)
        writer.add_scalar('losses_D', losses_D[i], i)
        writer.add_scalar('losses_G', losses_G[i], i)
        writer.add_scalar('losses_W', losses_W[i], i)

        if i%10==0:    
            torch.save(G.state_dict(), logging['G_save_path']+f"generator_epoch{i}.pt")
            torch.save(D.state_dict(), logging['D_save_path']+f"discriminator_epoch{i}.pt")

        if i%25==0:
            with torch.no_grad():
                for j, data in enumerate(val_data):

                    x = data[0].to(device).float()
                    c = data[1].to(device).float()
                    # if x.shape[0]<5: continue
                    
                    x = x.reshape(x.shape[0], x.shape[1], -1)
                    c = c.reshape(c.shape[0], c.shape[1], -1)

                    x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)

                    # x_syn (32, 300, 165), x (32, 300, 165), c (32, 300, 165)
                    print(f'plotting validation for epoch {i}')
                    vis_results(c, x, x_syn, i, logging['vis_test_results_path'], is_smplx=True)
                    break
                
                for j,data in enumerate(train_data):
                    x = data[0].to(device).float()
                    c = data[1].to(device).float()
                    # if x.shape[0]<5: continue
                    
                    x = x.reshape(x.shape[0], x.shape[1], -1)
                    c = c.reshape(c.shape[0], c.shape[1], -1)

                    x_syn = G(grf.sample(x.shape[0], x.shape[1]).float(),c)
                    
                    # x_syn (32, 300, 165), x (32, 300, 165), c (32, 300, 165)
                    print(f'plotting train for epoch {i}')
                    vis_results(c, x, x_syn, i, logging['vis_train_results_path'], is_smplx=True)

                    # del 
                    # gc 
                    # cuda cchche clean
                    
                    break
                
                

        
        with open(logging['log_loss_path'], 'a') as f:
            print("Epochs ",i, "D: ", losses_D[i], "G: ", losses_G[i], "W", losses_W[i], "mean: ", x.mean().item(), "std: ", x.std().item(), "losses MSE", losses_MSE[i], "losses_V_MSE", losses_V_MSE[i], file=f)

        
    return losses_D, losses_G, losses_W, losses_MSE, losses_V_MSE