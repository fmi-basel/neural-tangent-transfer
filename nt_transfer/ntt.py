import time
import itertools

import os

from jax.api import jit, grad

import jax.numpy as np

from jax import random

import random as orig_random

from nt_transfer.utils import *

from datetime import datetime

from datetime import timedelta  

import sys

from nt_transfer.nn_models import model_dict

from nt_transfer.load_datasets import Dataset

import logging

logger=logging.getLogger(__name__) 

out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)

logger.addHandler(out_hdlr)


class nt_transfer_model():
    def __init__(self, dataset_str, model_str, instance_input_shape, NN_DENSITY_LEVEL_LIST, OPTIMIZER_STR, NUM_RUNS, NUM_EPOCHS, BATCH_SIZE, STEP_SIZE, MASK_UPDATE_FREQ, LAMBDA_KER_DIST, LAMBDA_L2_REG, MASK_UPDATE_BOOL = True, VALIDATION_FRACTION = 0.002, PRUNE_METHOD = 'magnitude', GLOBAL_PRUNE_BOOL = False, INIT_RUN_INDEX = 1, SAVE_BOOL = True, save_dir = '/tungstenfs/scratch/gzenke/liutian/nt_transfer/saved_data/ntt_results/' ):
        """ 
        Args: 
            # Model options    
            dataset_str: a string that describes the dataset
            model_str: a string that describes the model
            instance_input_shape: the shape of a single input data sample
            NN_DENSITY_LEVEL_LIST: a list of desired weight density levels
            
            
            # Optimization options
            OPTIMIZER_STR: a string of optimizer
            NUM_RUNS: number of independent runs
            NUM_EPOCHS: number of epochs
            BATCH_SIZE: size of a minibatch
            STEP_SIZE: the learning rate
            MASK_UPDATE_FREQ: number of mini-batch iterations, after which mask will be updated according to the magnitudes of weights. 
            LAMBDA_KER_DIST: scaling parameter for kernel distance
            LAMBDA_L2_REG: scaling parameter for l2 penalty
            VALIDATION_FRACTION: the fraction of training data held-out as validation during NTT
            PRUNE_METHOD: the pruning method
            GLOBAL_PRUNE_BOOL: whether to use global (net-wise) pruning or not
            
            # Data & model saving options
            INIT_RUN_INDEX: the index of the initial run
            SAVE_BOOL: a boolean variable which decides whether the learned sparse parameters are saved.
            save_dir: the data saving directory.
        """
        
        self.model_str = model_str           
        self.NN_DENSITY_LEVEL_LIST = NN_DENSITY_LEVEL_LIST
        self.DATASET =  Dataset(datasource = dataset_str, VALIDATION_FRACTION = VALIDATION_FRACTION )        
        self.NUM_RUNS = NUM_RUNS
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_EPOCHS = NUM_EPOCHS
        self.OPTIMIZER_WITH_PARAMS = optimizer_dict[OPTIMIZER_STR](step_size = STEP_SIZE)
        self.LAMBDA_KER_DIST = LAMBDA_KER_DIST
        self.LAMBDA_L2_REG = LAMBDA_L2_REG
        self.SAVE_BOOL = SAVE_BOOL   
        self.INIT_RUN_INDEX = INIT_RUN_INDEX
        self.GLOBAL_PRUNE_BOOL = GLOBAL_PRUNE_BOOL
        self.PRUNE_METHOD = PRUNE_METHOD
        
        if MASK_UPDATE_BOOL == False:
            logger.info("Mask update during NTT has been turned off")
            self.MASK_UPDATE_FREQ = np.inf # not updating masks
        else:
            self.MASK_UPDATE_FREQ = MASK_UPDATE_FREQ # updating masks every a MASK_UPDATE_FREQ number of iterations        
        
#         time.sleep(orig_random.uniform(1,20))            
#         now = datetime.now()
#         now_str = str(now.strftime("%D:%H:%M:%S")).replace('/', ':')
        

        if GLOBAL_PRUNE_BOOL:
            global_layerwise_str = 'global_prune'
        else:
            global_layerwise_str = 'layerwise_prune'
                                            
#         self.unique_model_dir =  save_dir + dataset_str + '_' + global_layerwise_str + '_' + self.model_str + '__' + now_str
        self.unique_model_dir =  save_dir + dataset_str + '_' + global_layerwise_str + '_' + self.model_str 
        
        self.param_dict = dict(model_str = model_str, 
                               dataset_str = dataset_str, 
                               instance_input_shape = instance_input_shape,
                               NN_DENSITY_LEVEL_LIST = NN_DENSITY_LEVEL_LIST, 
                               OPTIMIZER_STR = OPTIMIZER_STR,
                               NUM_RUNS = NUM_RUNS, 
                               NUM_EPOCHS = NUM_EPOCHS, 
                               BATCH_SIZE = BATCH_SIZE,  
                               STEP_SIZE = STEP_SIZE, 
                               MASK_UPDATE_FREQ = self.MASK_UPDATE_FREQ, 
                               LAMBDA_KER_DIST =  LAMBDA_KER_DIST, 
                               LAMBDA_L2_REG = LAMBDA_L2_REG,
                               SAVE_BOOL = SAVE_BOOL,
                               VALIDATION_FRACTION = VALIDATION_FRACTION,
                               GLOBAL_PRUNE_BOOL = GLOBAL_PRUNE_BOOL,
                               PRUNE_METHOD = PRUNE_METHOD)  
        
        # unpack the neural net architecture
        init_fun, apply_fn = model_dict[model_str](W_initializers_str = 'glorot_normal()', b_initializers_str = 'normal()')
        
        self.init_fun = init_fun
        self.apply_fn = apply_fn
        self.emp_ntk_fn = empirical_ntk_fn(apply_fn)
        
        self.batch_input_shape = [-1] + instance_input_shape
    

        self.vali_samples = self.DATASET.dataset['val']['input'][:self.BATCH_SIZE, :].reshape(self.batch_input_shape)

        # split validation inputs into two collections, vali_inputs_1 and vali_inputs_2.
        half_vali_size = int(len(self.vali_samples)/2)
        self.vali_inputs_1 = self.vali_samples[:half_vali_size]
        self.vali_inputs_2 = self.vali_samples[half_vali_size:]


    def kernel_dist_target_dist_l2_loss(self, student_ker_mat, student_pred, teacher_ker_mat, teacher_pred, masked_params):

        """ Compute kernel distance, target distance, and parameter l2 loss.

        Args: 
            student_ker_mat: a student-network ntk matrix, 
            student_pred: a student-network prediction matrix.

            teacher_ker_mat: a teacher-network ntk matrix. 
            teacher_pred: a teacher-network prediction matrix. 

        Returns:
            ker_dist: squared l2 difference between two kernel matrices, normalized by the size of the matrix.
            target_dist: squared l2 norm difference between two prediction matrices, normalized by the size of the matrix.
            params_norm_squared: squared l2 norm of parameters.
        """ 
        
        # the normalized squared difference between teacher and student NTK matrices
        ker_dist = np.sum(np.square(student_ker_mat - teacher_ker_mat)) / teacher_ker_mat.size

        # the normalized squared difference between teacher and student network predictions
        target_dist = np.sum(np.square(student_pred - teacher_pred)) / student_pred.size

        # squared norm of parameters
        params_norm_squared = stax_params_l2_square(masked_params)

        return  ker_dist, target_dist, params_norm_squared 
    
        
    def eval_nt_transfer_loss_on_vali_data(self, masked_student_net_params, vali_teacher_prediction, vali_teacher_ntk_mat, density_level):
        """ Evaluate the ntk transfer loss using validation data.
         
         Args: 
            masked_student_net_params: the masked student network's parameters. 
            vali_teacher_prediction: the teacher network's prediction evaluated using validation inputs
            vali_teacher_ntk_mat: the teacher network's ntk matrix evaluated using validation inputs

        Returns:
            transfer_loss: the transfer loss
            ker_dist: the kernel distance
            target_dist: the target distance
            param_squared_norm: the l2 square of the parameters.
        """
        
        # evaluate the student ntk matrix using validation data.
        vali_student_ntk_mat = self.emp_ntk_fn(self.vali_inputs_1, self.vali_inputs_2, masked_student_net_params)   

        # evaluate the student prediction using validation data.        
        vali_student_prediction = self.apply_fn(masked_student_net_params, self.vali_samples) 

        # calculate the kernel distance, target distance, and parameter l2 loss
        ker_dist, target_dist, param_squared_norm = self.kernel_dist_target_dist_l2_loss(vali_student_ntk_mat, vali_student_prediction, vali_teacher_ntk_mat, vali_teacher_prediction, masked_student_net_params )
        
        # weight these distances and sum them up.
        weighted_ker_dist = self.LAMBDA_KER_DIST * ker_dist
        
        weighted_parameters_squared_norm = (self.LAMBDA_L2_REG / density_level) * param_squared_norm 

        transfer_loss = weighted_ker_dist +  target_dist + weighted_parameters_squared_norm

        return transfer_loss, ker_dist, target_dist, param_squared_norm


    def nt_transfer_loss(self, student_net_params, masks, teacher_net_params, x, density_level):

        """ The loss function of NTK transfer.

        Args: 
            student_net_params: network parameters. 
            masks: a collection of binary masks.
            apply_fn: a function that maps a tupe of network-parameters and a set of network-input to network-output.
            analytic_ntk_fn: an analytic (parameter independent) ntk function. It is the ntk of the teacher neural network. 
            emp_ntk_fn: an empirical (parameter dependent) ntk function. It is the ntk of the student neural network.
            teacher_net_params: a set of parameters used in dense, teacher network.
            x: the network inputs.

        Returns:
            transfer_loss: the loss function output.
        """ 

        masked_student_net_params = get_sparse_params_filtered_by_masks(student_net_params, masks)

        # split inputs into two collections, x1 and x2.
        x1 = x[:int(len(x)/2)]
        x2 = x[int(len(x)/2):]
        
        # student network prediction
        student_prediction = self.apply_fn(masked_student_net_params, x) 
        
        # teacher network prediction
        teacher_prediction = self.apply_fn(teacher_net_params, x)

        # student network's NTK evaluated on x1 and x2
        student_ntk_mat = self.emp_ntk_fn(x1, x2, masked_student_net_params)   

        # teacher network's NTK evaluated on x1 and x2
        teacher_ntk_mat = self.emp_ntk_fn(x1, x2, teacher_net_params)            

        # compute kernel, target, and paramter l2 loss
        ker_dist, target_dist, param_squared_norm = self.kernel_dist_target_dist_l2_loss(student_ntk_mat, student_prediction, teacher_ntk_mat, teacher_prediction, masked_student_net_params)

        # weight these losses to get the transfer loss
        transfer_loss = self.LAMBDA_KER_DIST * ker_dist +  target_dist + (self.LAMBDA_L2_REG / density_level) * param_squared_norm 

        return transfer_loss


    def optimize(self, return_teacher_params_bool = False):    
        """  Carry out the optimization task.
        Arg:
            run_index: the index of independent run of the optimization.
            save_dir: the directory used to save the transferred results.
            
        Returns:
            nt_trans_params_all_runs: the transferred parameters
            nt_trans_masks_all_runs: the transferred masks
            nt_trans_vali_all_runs: a collection o fvalidation loss during training.
        """

        gen_batches = self.DATASET.data_stream(self.BATCH_SIZE)
        
        num_complete_batches, leftover = divmod(self.DATASET.num_example['train'], self.BATCH_SIZE)

        # number of minibatches per epoch
        num_mini_batches_per_epochs = num_complete_batches +  bool(leftover)

        # number of total iterations
        num_total_iters  = self.NUM_EPOCHS * num_mini_batches_per_epochs

        # number of time that the sparisty levels get updated
        num_sparsity_updates = num_total_iters // self.MASK_UPDATE_FREQ 
     
        mask_update_limit =  num_total_iters - self.MASK_UPDATE_FREQ
    
        if self.SAVE_BOOL == True:
            # save the transferred results in the desinated directory.

            trans_model_dir = self.unique_model_dir

#             while os.path.exists(trans_model_dir):
#                 trans_model_dir = trans_model_dir + '_0'
            
            os.makedirs(trans_model_dir)

            np.save(trans_model_dir + '/param_dict.npy', self.param_dict) 
            
            

        nt_trans_params_all_sparsities_all_runs = []
        nt_trans_masks_all_sparsities_all_runs = []
        nt_trans_vali_all_sparsities_all_runs = []
        teacher_params_all_sparsities_all_runs = []
        
        
        num_sparisty_levels = len(self.NN_DENSITY_LEVEL_LIST) 
        num_runs = len(range(self.INIT_RUN_INDEX, self.INIT_RUN_INDEX + self.NUM_RUNS ))
        all_density_all_run_num_total_iters = num_sparisty_levels * num_runs * num_total_iters
        
        
        for nn_density_level in self.NN_DENSITY_LEVEL_LIST:   
            
            
            nt_trans_params_all_runs = []
            nt_trans_masks_all_runs = []
            nt_trans_vali_all_runs = []
            teacher_params_all_runs = []


            for run_index in range(self.INIT_RUN_INDEX, self.INIT_RUN_INDEX + self.NUM_RUNS ):

                # do logging
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                # a string that summarizes the current ntt experiment
                model_summary_str =  self.model_str + '_density_' + str(round(nn_density_level, 2) ) + '_run_' + str(run_index)

                if self.SAVE_BOOL == True:
                    model_dir_density_run = trans_model_dir + '/' + 'density_' + str(round(nn_density_level, 2) ) + '/' + 'run_' +  str(run_index) + '/'

                    os.makedirs(model_dir_density_run)
                    
                    logging.basicConfig(filename = model_dir_density_run + "/" + model_summary_str + "_log.log", format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG)

                else: 
                    logging.basicConfig(filename = model_summary_str + "_log.log" , format='%(asctime)s %(message)s', filemode='w', level=logging.DEBUG)
                
                
                # for different run indices, randomly draw teacher net's parameters
                _, teacher_net_params = self.init_fun(random.PRNGKey(run_index), tuple(self.batch_input_shape))
                                
                # the prediction of the teacher net evaluated on validation samples
                vali_teacher_prediction = self.apply_fn(teacher_net_params, self.vali_samples)

                vali_teacher_ntk_mat = self.emp_ntk_fn(self.vali_inputs_1, self.vali_inputs_2, teacher_net_params) 

                # the initial binary mask
                
                if self.PRUNE_METHOD == 'magnitude':                
                    masks = get_masks_from_jax_params(teacher_net_params, nn_density_level, global_bool = self.GLOBAL_PRUNE_BOOL)
                elif self.PRUNE_METHOD == 'logit_snip':
                    logger.info("Use logit snip method to get the initial mask")
                    num_examples_snip = 128

#                     gen_batches_logit_snip = self.DATASET.data_stream(num_examples_snip)
                    
                    snip_input = self.DATASET.dataset['train']['input'][:num_examples_snip, :]
                    
                    if self.GLOBAL_PRUNE_BOOL == False:
                        logger.warning("layerwise sparse net initialized with logit_snip")                        
                    masks = get_logit_snip_masks(teacher_net_params, nn_density_level, self.apply_fn, snip_input, self.batch_input_shape, GlOBAL_PRUNE_BOOL = self.GLOBAL_PRUNE_BOOL) 
                else:
                    raise NotImplementedError("not implemented")
    

                # the initial student parameters
                masked_student_net_params = get_sparse_params_filtered_by_masks(teacher_net_params, masks)

                # instantiate the optimizer triple 
                opt_init, opt_update, get_params = self.OPTIMIZER_WITH_PARAMS

                opt_state = opt_init(teacher_net_params) 

                # one step of NTK transfer
                @jit
                def nt_transfer_step(i, opt_state, x, masks):

                    # parameters in the current optimizer state
                    student_net_params = get_params(opt_state)

                    # gradients that flow through the binary masks
                    masked_g = grad(self.nt_transfer_loss)(student_net_params, masks, teacher_net_params, x, nn_density_level)

                    return opt_update(i, masked_g, opt_state)

                # a list of validation loss
                vali_loss_list = []

                # calculate the initial validation loss. 
                vali_loss = self.eval_nt_transfer_loss_on_vali_data(masked_student_net_params, vali_teacher_prediction, vali_teacher_ntk_mat, nn_density_level)

                vali_loss_list.append(vali_loss)

                logger.info("Before transfer: trans dist %.3f | ntk dist %.3f | targ dist %.3f | l2 pentalty %.3f | nn density %.2f", vali_loss[0], vali_loss[1], vali_loss[2], vali_loss[3], nn_density_level)
                itercount = itertools.count()

                t = time.time()

                # loop through iterations
                for num_iter in range(1, num_total_iters + 1): 
                    
                    # a batch of input data
                    batch_xs, _ = next(gen_batches)                

                    # reshape the input to a proper format (2d array for MLP and 3d for CNN)
                    batch_xs = batch_xs.reshape(self.batch_input_shape) 

                    # update the optimizer state
                    opt_state = nt_transfer_step(next(itercount), opt_state, batch_xs, masks )


                    if num_iter % 100 == 0:
                        elapsed_time = time.time() - t
                        
                        if (num_iter <= 500) and (run_index == self.INIT_RUN_INDEX) and (nn_density_level == self.NN_DENSITY_LEVEL_LIST[0]):  
                            # estimate the program end time.
                            remaining_iter_num = all_density_all_run_num_total_iters - num_iter
                            remaining_seconds = elapsed_time * ( remaining_iter_num / 100 )
                            expected_end_time = str(datetime.now() + timedelta(seconds = remaining_seconds))

                        # get parameters from the current optimizer state
                        student_net_params = get_params(opt_state) 

                        # filter the paramters by masks
                        masked_student_net_params = get_sparse_params_filtered_by_masks(student_net_params , masks)
                        
                        # validation loss
                        vali_loss = self.eval_nt_transfer_loss_on_vali_data(masked_student_net_params, vali_teacher_prediction, vali_teacher_ntk_mat, nn_density_level)    

                        vali_loss_list.append(vali_loss)

                        logger.info('run: %02d/%02d | iter %04d/%04d | trans. dist %.3f | ntk dist %.3f | targ. dist %.3f | l2 %.3f | nn density %.2f | time %.2f [s] | expected finish time %s', run_index, self.NUM_RUNS + self.INIT_RUN_INDEX - 1, num_iter, num_total_iters, vali_loss[0], vali_loss[1], vali_loss[2], vali_loss[3], nn_density_level, elapsed_time, expected_end_time)
                        t = time.time()


                    if (num_iter % self.MASK_UPDATE_FREQ == 0) and (num_iter < mask_update_limit):
                        # get parameters from the current optimizer state
                        student_net_params = get_params(opt_state) 
                        
                        # update masks
                        masks = get_masks_from_jax_params(student_net_params, nn_density_level, global_bool = self.GLOBAL_PRUNE_BOOL)
                        
#                         if self.PRUNE_METHOD == 'logit_snip':
#                             logit_snip_batch_xs, _ = next(gen_batches_logit_snip)
#                             masks = get_logit_snip_masks(student_net_params, nn_density_level, self.apply_fn, snip_input, self.batch_input_shape, GlOBAL_PRUNE_BOOL = self.GLOBAL_PRUNE_BOOL) 
#                         else:
#                             masks = get_masks_from_jax_params(student_net_params, nn_density_level, global_bool = self.GLOBAL_PRUNE_BOOL)


                
                elapsed_time = time.time() - t
                
                student_net_params = get_params(opt_state) 
                
                # filter the paramters by masks
                masked_student_net_params = get_sparse_params_filtered_by_masks(student_net_params , masks)
                
                vali_loss = self.eval_nt_transfer_loss_on_vali_data(masked_student_net_params, vali_teacher_prediction, vali_teacher_ntk_mat, nn_density_level)    

                vali_loss_list.append(vali_loss)
                
                logger.info('run: %02d/%02d | iter %04d/%04d | trans. dist %.3f | ntk dist %.3f | targ. dist %.3f | l2 %.3f | nn density %.2f | time %.2f [s]', run_index, self.NUM_RUNS + self.INIT_RUN_INDEX - 1, num_iter, num_total_iters, vali_loss[0], vali_loss[1], vali_loss[2], vali_loss[3], nn_density_level, elapsed_time )
                            
                vali_loss_array = np.array(vali_loss_list)

                nt_trans_params_all_runs.append(masked_student_net_params)
                nt_trans_masks_all_runs.append(masks)
                nt_trans_vali_all_runs.append(vali_loss_array)
                teacher_params_all_runs.append(teacher_net_params )

                if self.SAVE_BOOL == True:

                    model_summary_str =  self.model_str + '_density_' + str(round(nn_density_level, 2) ) + '_run_' + str(run_index)

                    teacher_param_fileName = model_dir_density_run + 'teacher_params_' + model_summary_str
                    np.save(teacher_param_fileName, teacher_net_params)

                    student_param_fileName = model_dir_density_run + 'transferred_params_' + model_summary_str
                    np.save(student_param_fileName, masked_student_net_params)

                    mask_fileName = model_dir_density_run + 'transferred_masks_' + model_summary_str
                    np.save(mask_fileName, masks)

                    loss_array_fileName = model_dir_density_run + 'loss_array_' + model_summary_str
                    np.save(loss_array_fileName, vali_loss_array)
            

        nt_trans_params_all_sparsities_all_runs.append( nt_trans_params_all_runs )
        nt_trans_masks_all_sparsities_all_runs.append( nt_trans_masks_all_runs )
        nt_trans_vali_all_sparsities_all_runs.append( nt_trans_vali_all_runs )
        teacher_params_all_sparsities_all_runs.append( teacher_params_all_runs )
                    
        if return_teacher_params_bool:
            return nt_trans_params_all_sparsities_all_runs, nt_trans_masks_all_sparsities_all_runs, nt_trans_vali_all_sparsities_all_runs, teacher_params_all_sparsities_all_runs

        else:
            return nt_trans_params_all_sparsities_all_runs, nt_trans_masks_all_sparsities_all_runs, nt_trans_vali_all_sparsities_all_runs
    