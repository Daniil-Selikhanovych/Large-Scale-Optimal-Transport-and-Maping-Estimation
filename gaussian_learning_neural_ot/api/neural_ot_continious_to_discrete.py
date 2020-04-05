import torch
import time

from .gaussian_functions import index_sampler

from .constants import (f_net_default, u_net_default, v_vec_default,
			c_cost_type_default, d_cost_type_default,
			reg_modes_arr,
			reg_mode_default, eps_default,
			epochs_default, batch_size_default, 
			dtype_default, device_default,
			random_state_default, random_states_train_default,
			mu_sampler_default, data_nu_val_default, 
			optimizer_mode_default, lr_default)
			
from .neural_ot import Neural_OT

class Neural_OT_continious_to_discrete(Neural_OT):
    
    def __init__(self, f_net = f_net_default, u_net = u_net_default, v_vec = v_vec_default, 
                 reg_mode = reg_mode_default, eps = eps_default, 
                 dtype = dtype_default, device = device_default):

        Neural_OT.__init__(self, f_net = f_net, reg_mode = reg_mode, eps = eps, 
                 c_cost_type = c_cost_type_default,
                 d_cost_type = d_cost_type_default,
                 dtype = dtype_default, device = device_default)

        self.u = u_net.to(device)
        self.v = v_vec.to(device)
        
    def replace_u(self, u):
        self.u = u.to(self.device)
        
    def replace_v(self, v):
        self.v = v.to(self.device)
        
    def stochastic_OT_computation(self, epochs = epochs_default, batch_size = batch_size_default,
                                  random_state_val = random_state_default,
                                  random_states_train = random_states_train_default,
                                  mu_sampler = mu_sampler_default, 
                                  index_sampler = index_sampler,
                                  nu_data = data_nu_val_default,
                                  optimizer_mode = optimizer_mode_default, 
                                  lr = lr_default,
                                  loss_arr_batch = [],
                                  loss_arr_val = []):
        if (self.v.shape[0] != nu_data.shape[0]):
            raise ValueError("Vector v and nu_data should be the same size!")

        trainable_params = list(self.u.parameters()) + [self.v]
        
        if optimizer_mode == 'Adam':
            optimizer = torch.optim.Adam(trainable_params, lr = lr)
        elif optimizer_mode == 'SGD':
            optimizer = torch.optim.SGD(trainable_params, lr = lr)

        for epoch in range(epochs):

            start_time = time.time()
            
            x_batch = mu_sampler(random_state = random_states_train[epoch], batch_size = batch_size)
            #print(x_batch.device)
            
            indexes_to_choice = index_sampler(nu_data_shape = nu_data.shape[0], 
                                              batch_size = batch_size, 
                                              random_state = random_states_train[epoch], 
                                              device = self.device)
            y_batch = nu_data[indexes_to_choice, :]
            u_batch = (self.u)(x_batch)
            v_batch = (self.v)[indexes_to_choice]
            
            loss_batch = self.dual_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch)
            
            optimizer.zero_grad()

            loss_batch.backward()
            optimizer.step()


            end_time = time.time()
            consumed_time = end_time - start_time

            loss_batch_maximization = -loss_batch.item()
            
            x_batch_val = mu_sampler(random_state = random_state_val, batch_size = nu_data.shape[0])
            y_batch_val = nu_data
            
            self.u.eval()
            self.v.requires_grad_(False)
            
            u_batch_val = (self.u)(x_batch_val)
            v_batch_val = self.v
            
            loss_val = self.dual_OT_loss_estimation(u_batch_val, v_batch_val, 
                                                         x_batch_val, y_batch_val)
            
            loss_val_maximization = -loss_val.item()
            
            if (epoch % 50 == 0):
            	print("------------------------------")
            	print(f"Epoch_num = {epoch + 1}")
            	print(f"Consumed time = {consumed_time} seconds")
            	print(f"Loss estimation on sampled data = {loss_batch_maximization}")
            	print(f"Loss estimation on validation data = {loss_val_maximization}")

            loss_arr_batch.append(loss_batch_maximization)
            loss_arr_val.append(loss_val_maximization)
        
    def optimal_map_learning(self, epochs = epochs_default, batch_size = batch_size_default,
                                  random_state_val = random_state_default,
                                  random_states_train = random_states_train_default,
                                  mu_sampler = mu_sampler_default, 
                                  index_sampler = index_sampler,
                                  nu_data = data_nu_val_default,
                                  optimizer_mode = optimizer_mode_default, 
                                  lr = lr_default,
                                  loss_arr_batch = [],
                                  loss_arr_val = []):
        
        trainable_params = list(self.f_net.parameters())
        
        if optimizer_mode == 'Adam':
            optimizer = torch.optim.Adam(trainable_params, lr = lr)
        elif optimizer_mode == 'SGD':
            optimizer = torch.optim.SGD(trainable_params, lr = lr)
            

        for epoch in range(epochs):

            start_time = time.time()
            
            x_batch = mu_sampler(random_state = random_states_train[epoch], batch_size = batch_size)
            #print(x_batch.device)
            
            indexes_to_choice = index_sampler(nu_data_shape = nu_data.shape[0], 
                                              batch_size = batch_size, 
                                              random_state = random_states_train[epoch], 
                                              device = self.device)
            y_batch = nu_data[indexes_to_choice, :]
            u_batch = (self.u)(x_batch)
            v_batch = (self.v)[indexes_to_choice]
            
            self.f_net.train()
            map_batch = (self.f_net)(x_batch)
            
            loss_batch = self.mapping_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch, map_batch)
            
            optimizer.zero_grad()

            loss_batch.backward()
            optimizer.step()


            end_time = time.time()
            consumed_time = end_time - start_time

            loss_batch = loss_batch.item()
            
            x_batch_val = mu_sampler(random_state = random_state_val, batch_size = nu_data.shape[0])
            y_batch_val = nu_data
            
            u_batch_val = (self.u)(x_batch_val)
            v_batch_val = self.v
            
            self.f_net.eval()
            map_batch = (self.f_net)(x_batch)
            
            loss_val = self.mapping_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch, map_batch)
            
            loss_val = loss_val.item()
            if (epoch % 50 == 0):
            	print("------------------------------")
            	print(f"Epoch_num = {epoch + 1}")
            	print(f"Consumed time = {consumed_time} seconds")
            	print(f"Loss estimation on sampled data = {loss_batch}")
            	print(f"Loss estimation on validation data = {loss_val}")

            loss_arr_batch.append(loss_batch)
            loss_arr_val.append(loss_val)
            
    def optimal_map_learning_algo_2(self, epochs = epochs_default, batch_size = batch_size_default,
                                  random_state_val = random_state_default,
                                  random_states_train = random_states_train_default,
                                  mu_sampler = mu_sampler_default, 
                                  index_sampler = index_sampler,
                                  nu_data = data_nu_val_default,
                                  lr = lr_default,
                                  loss_arr_batch = [],
                                  loss_arr_val = []):
        
        for epoch in range(epochs):

            x_batch = mu_sampler(random_state = random_states_train[epoch], batch_size = batch_size)
            
            indexes_to_choice = index_sampler(nu_data_shape = nu_data.shape[0], 
                                              batch_size = batch_size, 
                                              random_state = random_states_train[epoch], 
                                              device = self.device)
            y_batch = nu_data[indexes_to_choice, :]
            u_batch = (self.u)(x_batch)
            v_batch = (self.v)[indexes_to_choice]
            
            start_time = time.time()
            
            self.f_net.zero_grad()
            #self.f_net.train()
            map_batch = (self.f_net)(x_batch)
            loss_batch = self.mapping_OT_loss_estimation(u_batch, v_batch, x_batch, y_batch, map_batch)
            
            loss_batch.backward()
            #data_nu = data_nu.to(device)
            #data_mu = data_mu.to(device)

            f_params_dict = {params_name: params for params_name, params in zip(self.f_net.state_dict(), 
                                                                             self.f_net.parameters())}
            
            f_grad_dict = {params_name: params.grad*lr
                               for params_name, params in zip(self.f_net.state_dict(), self.f_net.parameters())}
            
            for params_name, params in self.f_net.state_dict().items():
                self.f_net.state_dict()[params_name].data.copy_(f_params_dict[params_name] - \
                                                           f_grad_dict[params_name])

            end_time = time.time()
            consumed_time = end_time - start_time

            loss_batch = loss_batch.item()
            
            x_batch_val = mu_sampler(random_state = random_state_val, batch_size = nu_data.shape[0])
            y_batch_val = nu_data
            
            u_batch_val = (self.u)(x_batch_val)
            v_batch_val = self.v
            
            self.f_net.eval()
            map_batch_val = (self.f_net)(x_batch_val)
            
            loss_val = self.mapping_OT_loss_estimation(u_batch_val, v_batch_val, 
                                                       x_batch_val, y_batch_val, map_batch_val)
            
            loss_val = loss_val.item()
            
            if (epoch % 50 == 0):
            	print("------------------------------")
            	print(f"Epoch_num = {epoch + 1}")
            	print(f"Consumed time = {consumed_time} seconds")
            	print(f"Loss estimation on sampled data = {loss_batch}")
            	print(f"Loss estimation on validation data = {loss_val}")

            loss_arr_batch.append(loss_batch)
            loss_arr_val.append(loss_val)
