
import torch
import math
from torch.autograd import Variable
import numpy as np

from pyhessian.utils import get_params_grad_with_print, group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal
import collections 
from functools import partial



class hessian_with_activation():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion
        self.activations = collections.defaultdict(list)
        self.activation_grads = collections.defaultdict(list)

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def insert_hook(self, module_name):
        def save_input(name, module, input, output):
            self.activations[name].append(input[0])
        
        def save_input_grad(name, module, in_grad, out_grad):
            self.activation_grads[name].append(in_grad[0])

        for name, m in self.model.named_modules():
            if not name.endswith(module_name):
                continue
            m.register_forward_hook(partial(save_input, name))
            m.register_backward_hook(partial(save_input_grad, name))
            # print(f"{name} module hooked")

    def check_reg_hook_size(self):
        for layer in self.activations.keys():
            input_size = torch.randint_like(self.activations[layer], high=2, device="cuda").size()
            if self.activation_grads[layer] is not None:
                grad_size = self.activation_grads[layer].size()
            else:
                grad_size = 1
            # print(type(self.activation_grads[layer][i][0]))
            if(input_size != grad_size):
                print(f"########## {layer} not equal !! #######")
                print( input_size, grad_size, "\n\n")
            else:
                print(f"************* {layer} ************")
                print( input_size, "\n\n")

    def get_activ_rand_v(self, show_layer=False, dont_reset=False):
        self.reset_reg_active()

        device = self.device
        for inputs, targets in self.data:
            break
        self.model.zero_grad()
        # print(f"********* 1 iteration input size : {len(inputs)}")
        outputs = self.model(inputs.to(device))
        loss = self.criterion(outputs, targets.to(device))
        loss.backward(create_graph=True)

        activs, _ = self.get_same_size_activ_grad(show_layer=show_layer, dont_reset=dont_reset)

        v = [
            torch.randint_like(p, high=2, device=device)
            for p in activs
        ]
        self.model.zero_grad()
        if dont_reset is not True:
            self.reset_reg_active()
        return v

    def get_same_size_activ_grad(self, show_layer=False, dont_reset=False):
        # rand_vs = []
        activ_grads = []
        activs = []
        for layer in self.activations.keys():
            activ_element = self.activations[layer][0]
            grad_element = self.activation_grads[layer][0]

            if len(self.activations[layer]) != 1:
                print(len(self.activations[layer]), self.activations[layer][0].size())
                raise Exception("register hook have several batchs.\
                     In processing hessian activation, register hook must have only one batch")

            if (grad_element is None):
                continue
            elif grad_element.size() != activ_element.size() :
                continue
            else:    
                # rand_vs.append(torch.randint_like(self.activations[layer], high=2, device="cuda"))
                if show_layer:
                    print(f"append {layer}, active size : {activ_element.size()}, active grad size : {grad_element.size()}")
                activ_grads.append(grad_element)
                activs.append(activ_element)
        if dont_reset is not True:
            self.reset_reg_active()
        return activs, activ_grads

    def reset_reg_active(self):
        for layer in self.activations.keys():
            self.activations[layer] = []
            self.activation_grads[layer] = []

    def dataloader_activation_hv_product(self, active_v):
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        Active_THv = [torch.zeros(p.size()).to(device) for p in active_v]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()

            self.reset_reg_active()

            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)

            activations, activation_grads = self.get_same_size_activ_grad()

            self.model.zero_grad()
            
            if active_v[0].size(0) == inputs.size(0) :
                Active_Hv = []
                for (i, (grad, act, v)) in enumerate(zip(activation_grads, activations, active_v)):

                    Active_Hv_element = torch.autograd.grad(grad,
                                        act,
                                        grad_outputs=v,
                                        only_inputs=True,
                                        retain_graph=True)
                    Active_Hv.append(Active_Hv_element[0])
                tmp_num_data = inputs.size(0)
                Active_THv = [
                    THv1 + Hv1 * float(tmp_num_data) + 0.
                    for THv1, Hv1 in zip(Active_THv, Active_Hv)
                ]
                num_data += float(tmp_num_data)
            else:
                print("input and rand v has different batch size")

        Active_THv = [THv1 / float(num_data) for THv1 in Active_THv]
        eigenvalue = group_product(Active_THv, active_v).cpu().item()
        return eigenvalue, Active_THv

    def trace_activ(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """
        device = self.device
        Active_trace_vhv = []
        Active_trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            active_v = self.get_activ_rand_v()
            # generate Rademacher random variables
            for v_i in active_v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Active_Hv = self.dataloader_activation_hv_product(active_v)

            Active_trace_vhv.append([group_product(Hv, v).cpu().item() for Hv, v in zip(Active_Hv, active_v)])
            if abs(np.mean(Active_trace_vhv) - Active_trace) / (abs(Active_trace) + 1e-6) < tol:
                print(f"In {i}th iteration, trace had been converge")
                return Active_trace_vhv
            else:
                Active_trace = np.mean(Active_trace_vhv)
        
        print(f"trace had not been converge")
        return Active_trace_vhv


    def dataloader_hv_product(self, v):
        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params]  # accumulate result
        for inputs, targets in self.data:
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            loss = self.criterion(outputs, targets.to(device))
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv


    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """

        device = self.device
        trace_vhv = []
        trace = 0.

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_vhv.append(group_product(Hv, v).cpu().item())
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    
    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            ############### Lanczos
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]
            a_, b_ = torch.eig(T, eigenvectors=True)

            eigen_list = a_[:, 0]
            weight_list = b_[0, :]**2
            eigen_list_full.append(list(eigen_list.cpu().numpy()))
            weight_list_full.append(list(weight_list.cpu().numpy()))

        return eigen_list_full, weight_list_full
