from jax.api import jit, grad, eval_shape, jacobian, jvp, vjp

import jax.numpy as np

from jax import random

# from jax.experimental import optimizers

from jax.experimental.stax import logsoftmax

from functools import partial

import tempfile

import itertools as IT

import os


from jax.experimental import optimizers

optimizer_dict = {}
optimizer_dict['adam'] = optimizers.adam
optimizer_dict['sgd'] = optimizers.sgd
    
    
    
def accuracy(params, f, images, targets, key = None):
    target_class = np.argmax(targets, axis=1)
    
    if key is not None:
        predicted_class = np.argmax(f(params, images, rng = key), axis=1)
    else:
        predicted_class = np.argmax(f(params, images), axis=1)
    return np.mean(predicted_class == target_class)
                                    
    

# def mean_and_variation_acc_across_trials(results_mat):
#     """
#     Args:
#     results_mat: A `np.ndarray` of shape [n_trials, n_data_points_evaluated]

#     Returns:
#     av_results_mat_across_trials: averaged results across trials (axis 0).

#     max_min_diff_ntk_distance: the upper/lower limit results obtaiend in different trials centered acround the results of av_results_mat_across_trials

#     """
#     av_results_mat_across_trials =  np.float32(np.mean(results_mat, axis = 0))
    
#     max_min_in_trials = np.vstack( (np.amin(results_mat, axis = 0 ), np.amax(results_mat, axis = 0 ) ) ) 
    
#     max_min_centered_diff = np.abs(max_min_in_trials - av_results_mat_across_trials)

#     return av_results_mat_across_trials, max_min_centered_diff


def _read_keys(keys):
    if keys is None or (isinstance(keys, np.ndarray) and keys.shape == (2,)):
        key1 = key2 = keys
    elif isinstance(keys, tuple):
        # assuming x1 and x2 using key1 and key2, resp.
        key1, key2 = keys
    elif isinstance(keys, np.ndarray) and keys.shape == (2, 2):
        key1, key2 = keys[0], keys[1]
    else:
        raise ValueError('`keys` must be one of the following: `None`, a PRNG '
                     'key, a tuple of PRNG keys or a (2, 2) array and dtype '
                     'unint32')
    return key1, key2

def empirical_ntk_fn(f):
    """Computes the ntk without batching for inputs x1 and x2.
    The Neural Tangent Kernel is defined as J(X_1)^T J(X_2) where J is the
    jacobian df/dparams. Computing the NTK directly involves directly
    instantiating the jacobian which takes
    O(dataset_size * output_dim * parameters) memory. It turns out it is
    substantially more efficient (especially as the number of parameters grows)
    to compute the NTK implicitly.
    This involves using JAX's autograd to compute derivatives of linear functions
    (which do not depend on the inputs). Thus, we find it more efficient to refer
    to fx_dummy for the outputs of the network. fx_dummy has the same shape as
    the output of the network on a single piece of input data.
    TODO(schsam): Write up a better description of the implicit method.
    Args:
    f: The function whose NTK we are computing. f should have the signature
       f(params, inputs) and should return an `np.ndarray` of outputs with shape
       [|inputs|, output_dim].
    Returns:
    A function ntk_fn that computes the empirical ntk.
    """

    def ntk_fn(x1, x2, params, keys=None):
        """Computes the empirical ntk.
        Args:
          x1: A first `np.ndarray` of inputs, of shape [n1, ...], over which we
            would like to compute the NTK.
          x2: A second `np.ndarray` of inputs, of shape [n2, ...], over which we
            would like to compute the NTK.
          params: A PyTree of parameters about which we would like to compute the
            neural tangent kernel.
          keys: None or a PRNG key or a tuple of PRNG keys or a (2, 2) array and
            dtype uint32. If `key == None`, then the function `f` is deterministic
            and requires no PRNG key; else if `keys` is a single PRNG key, then x1
            and x2 must be the same and share the same PRNG key; else x1 and x2 use
            two different PRNG keys.
        Returns:
          A `np.ndarray` of shape [n1, n2] + output_shape + output_shape.
        """
        key1, key2 = _read_keys(keys)
        # TODO(xlc): find a good way to check utils.x1_is_x2(x1, x2) == (key1==key2)
        if x2 is None:
            x2 = x1

        f_dummy = partial(f, rng=random.PRNGKey(1))
        fx2_struct = eval_shape(f_dummy, params, x2)
        fx_dummy = np.ones(fx2_struct.shape, fx2_struct.dtype)
        def delta_vjp_jvp(delta):
            def delta_vjp(delta):
                return vjp(lambda p: f(p, x2, rng=key2), params)[1](delta)
            return jvp(lambda p: f(p, x1, rng=key1), (params,), delta_vjp(delta))[1]

        ntk = jacobian(delta_vjp_jvp)(fx_dummy)
        ndim = len(fx2_struct.shape)
        ordering = (0, ndim) + tuple(range(1, ndim)) + \
            tuple(x + ndim for x in range(1, ndim))
        return np.transpose(ntk, ordering)

    return ntk_fn

def get_masks_from_jax_params(params, nn_density_level, magnitude_base_bool = True, global_bool = False, reshuffle_seed = 0):
    """ Assemble a collection of 0-1 valued masks which are of the same sizes and shapes as layers' weight tensors
        Note that this function ignores bias parameters.
    
    Args: 
        params: parameters in a jax.experimental.stax format. 
        nn_density_level: the desired density level for weight parameters.
        magnitude_base_bool: a boolean variable that decides whether to prune the network by magnitude or randomly prune the network
        
    Returns:
        masks: a collection of 0-1 valued masks which are of the same sizes and shapes as the layers' weight tensors.
    """ 
    
    if (type(magnitude_base_bool) != bool) or (type(global_bool) != bool):
        raise ValueError("magnitude_base_bool and global_bool should be boolean variables")
    
    masks = []
    
    if global_bool:
        weight_magnitudes_pooled = np.concatenate([ np.abs(layer_params[0].flatten()) for layer_params in params if len(layer_params) > 1])
        idx = int( (1 - nn_density_level) * np.size(weight_magnitudes_pooled) )
        global_thres = np.sort(weight_magnitudes_pooled)[idx]
    
    for layer_index in range( len(params)):


        if len(params[layer_index]) < 2:
            # In this the case, the layer does not contain weight and bias parameters.
            masks.append( [] )
            
        elif len(params[layer_index]) == 2:
            # In this case, the layer contains a tuple of parameters for weights and biases
            
            weights = params[layer_index][0]
            
            weight_magnitudes = np.abs(weights)

            if global_bool and magnitude_base_bool:
                
                this_mask = np.float32(weight_magnitudes > global_thres)
                
            else:
                # index: number of pruned parameters
                idx = int( (1 - nn_density_level) * np.size(weights) )

                # threshold: entries which below the thredhold will be removed
                thres = np.sort(np.reshape(weight_magnitudes, [-1] ))[idx]

                # 0 selected for weight parameters with magnitudes smaller than the threshold, 1 otherwise
                this_mask = np.float32(weight_magnitudes > thres)

                if magnitude_base_bool == False:
                    # in the case of random pruning: randomly shuffle the mask
                    this_mask = random.shuffle(random.PRNGKey(0), this_mask ) 

            masks.append(this_mask ) 

        else:
            raise NotImplementedError

    return masks


def get_sparse_params_filtered_by_masks(params, masks):
    
    """ Filter jax.experimental.stax parameters according to a collection of masks
    
    Args: 
        params: parameters in a stax format. 
        masks: a collection of masks for parameter pruning.
        
    Returns:
        sparse_params: sparse parameters acquired by passing params through the binay masks. 
    """ 
        
    sparse_params = []
    for layer_index in range( len(params)):


        if len(params[layer_index]) < 2:
            # In this the case, the layer does not contain weight or bias parameters
            sparse_params.append( () )
            
        elif len(params[layer_index]) == 2:
            # In this case, the layer contains a tuple of parameters for weights and biases
            weights  = params[layer_index][0]
            
            biases = params[layer_index][1]
            
            mask_this_layer = masks[layer_index]
            
            # sparse weights gated by masks
            sparse_weights = np.multiply(mask_this_layer, weights )
            
            # weight-bias tuple
            sparse_params_this_layer = (sparse_weights, biases )
            
            sparse_params.append( sparse_params_this_layer ) 
        else:
            raise NotImplementedError


    return sparse_params


def stax_params_l2_square(params, mask = None, regularize_bias_bool = False):

        
    """ Compute squared l2 norm of stax parameters
    
    Args: 
        params: parameters in a stax format. 
        regularize_bias_bool: a boolean var deciding wether biases are regularized or not.
    Returns:
        params_norm_squared: squared l2 norm of stax parameters. 
    """         
    if (type(regularize_bias_bool) != bool):
        raise ValueError("regularize_bias_bool should be a boolean variable")        
    
    if regularize_bias_bool == False:
        # assemble a list of weight parmeters; discard bias parameters 
        list_params = [layer_param[0] for layer_param in params if len(layer_param) == 2 ]
    else:
        # remove the empty tuple from the list of paramters.
        list_params = list(sum(params, ()))     
    
    params_norm_squared = np.sum([ np.sum(np.square( list_params[i] ))  for i in range(len(list_params)) ]  )
            
    return params_norm_squared
    
def softmax_cross_entropy_with_logits_l2_reg(params, f, x, targets, masks = None, L2_REG_COEFF = 0.0, key = None): 

    """ cross entropy loss + weighted l2 regularization. 
   
    Args: 
        params: parameters in a stax format. 
        apply_fn: a function that maps a set of network-parameters together with a set of network-inputs to network-outputs.
        x: network inputs
        targets: the target outputs
        masks: the sparsity-inducing mask
        L2_REG_COEFF: l2 regularization constant.
    Returns:
        The cross entropy loss + weighted l2 regularization
    """ 
    
    if masks is not None:
        masked_params = get_sparse_params_filtered_by_masks(params, masks)
    else:
        masked_params = params
    
    if key is not None:
        dense_outputs = f(masked_params, x, rng = key)     
    else:
        dense_outputs = f(masked_params, x)
        
    preds = logsoftmax(dense_outputs)
    
    params_norm_squared = stax_params_l2_square(masked_params)
        
    return -np.mean(np.sum(preds * targets, axis=1)) + L2_REG_COEFF * params_norm_squared


def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    
# def get_snip_masks(params, nn_density_level, predict, snip_batch, batch_input_shape):

#     init_grads = grad(softmax_cross_entropy_with_logits_l2_reg)(params, predict, snip_batch[0].reshape(batch_input_shape), snip_batch[1]) 

#     cs = [abs( init_grads[idx][0] *  params[idx][0]).flatten() for idx in range(len(params)) if len(params[idx]) == 2 ]

#     pooled_cs = np.hstack(cs)

#     idx = int( (1 - nn_density_level) * len(pooled_cs) )

#     # threshold: entries which below the thredhold will be removed
#     thres = np.sort(pooled_cs)[idx]

#     masks = []
#     for layer_index in range( len(params)):

#         if len(params[layer_index]) < 2:
#             # In this the case, the layer does not contain weight and bias parameters.
#             masks.append( [] )

#         elif len(params[layer_index]) == 2:
#             # In this case, the layer contains a tuple of parameters for weights and biases

#             weights = params[layer_index][0]

#             weights_grad = init_grads[layer_index][0]

#             layer_cs = np.abs(weights * weights_grad)

#             # 0 selected for weight parameters with magnitudes smaller than the threshold, 1 otherwise
#             this_mask = np.float32(layer_cs >= thres)

#             masks.append(this_mask ) 

#         else:
#             raise NotImplementedError
            
#     return masks   
    
def get_snip_masks(params, nn_density_level, predict, snip_batch, batch_input_shape, GlOBAL_PRUNE_BOOL = True):

    init_grads = grad(softmax_cross_entropy_with_logits_l2_reg)(params, predict, snip_batch[0].reshape(batch_input_shape), snip_batch[1]) 

    thres_list = [None] * len(params)

    if GlOBAL_PRUNE_BOOL == True: # global pruning

        cs = [abs( init_grads[idx][0] *  params[idx][0]).flatten() for idx in range(len(params)) if len(params[idx]) == 2 ]

        pooled_cs = np.hstack(cs)

        idx = int( (1 - nn_density_level) * len(pooled_cs) )

        # threshold: entries which below the thredhold will be removed
        thres = np.sort(pooled_cs)[idx]
        thres_list = [thres] * len(params)
        
    else: # layerwise pruning
        for layer_index in range( len(params)):
            if len(params[layer_index]) == 2:

                cs = abs( init_grads[layer_index][0] *  params[layer_index][0]).flatten()
                idx = int( (1 - nn_density_level) * len(cs) )
                # threshold: entries which below the thredhold will be removed
                thres = np.sort(cs)[idx]            
                thres_list[layer_index] = thres


    masks = []
    for layer_index in range( len(params)):

        if len(params[layer_index]) < 2:
            # In this the case, the layer does not contain weight and bias parameters.
            masks.append( [] )

        elif len(params[layer_index]) == 2:
            # In this case, the layer contains a tuple of parameters for weights and biases

            weights = params[layer_index][0]

            weights_grad = init_grads[layer_index][0]

            layer_cs = np.abs(weights * weights_grad)

            # 0 selected for weight parameters with magnitudes smaller than the threshold, 1 otherwise
            this_mask = np.float32(layer_cs >= thres_list[layer_index])

            masks.append(this_mask ) 


        else:
            raise NotImplementedError
            
    return masks



def get_logit_snip_masks(params, nn_density_level, predict, x_batch, batch_input_shape, GlOBAL_PRUNE_BOOL = True):
    
    def norm_square_logits(params, f, x): 
        return np.sum(f(params, x) **2)
    
    init_grads = grad(norm_square_logits)(params, predict, x_batch.reshape(batch_input_shape) ) 

    thres_list = [None] * len(params)

    if GlOBAL_PRUNE_BOOL == True: # global pruning

        cs = [abs( init_grads[idx][0] *  params[idx][0]).flatten() for idx in range(len(params)) if len(params[idx]) == 2 ]

        pooled_cs = np.hstack(cs)

        idx = int( (1 - nn_density_level) * len(pooled_cs) )

        # threshold: entries which below the thredhold will be removed
        thres = np.sort(pooled_cs)[idx]
        thres_list = [thres] * len(params)
        
    else: # layerwise pruning
        for layer_index in range( len(params)):
            if len(params[layer_index]) == 2:

                cs = abs( init_grads[layer_index][0] *  params[layer_index][0]).flatten()
                idx = int( (1 - nn_density_level) * len(cs) )
                # threshold: entries which below the thredhold will be removed
                thres = np.sort(cs)[idx]            
                thres_list[layer_index] = thres

            

    masks = []
    for layer_index in range( len(params)):

        if len(params[layer_index]) < 2:
            # In this the case, the layer does not contain weight and bias parameters.
            masks.append( [] )

        elif len(params[layer_index]) == 2:
            # In this case, the layer contains a tuple of parameters for weights and biases

            weights = params[layer_index][0]

            weights_grad = init_grads[layer_index][0]

            layer_cs = np.abs(weights * weights_grad)

            # 0 selected for weight parameters with magnitudes smaller than the threshold, 1 otherwise
            this_mask = np.float32(layer_cs >= thres_list[layer_index])

            masks.append(this_mask ) 


        else:
            raise NotImplementedError
            
    return masks


def mean_and_var_across_trials(results_mat):
    """
    Args:
    results_mat: A `np.ndarray` of shape [n_trials, n_data_points_evaluated]

    Returns:
    av_results_mat_across_trials: averaged results across trials (axis 0).

    var_results_mat_across_trials: the variance across trials
    """
    
    av_results_mat_across_trials =  np.float32(np.mean(results_mat, axis = 0))
    var_results_mat_across_trials = np.var(results_mat, axis = 0)
#     max_min_in_trials = np.vstack( (np.amin(results_mat, axis = 0 ), np.amax(results_mat, axis = 0 ) ) ) 
    
#     max_min_centered_diff = np.abs(max_min_in_trials - av_results_mat_across_trials)

    return av_results_mat_across_trials, var_results_mat_across_trials


def load_supervised_training_result(supervised_result_file, wiring_str, nn_density_list, num_plots = -1, supervised_result_path = '/tungstenfs/scratch/gzenke/liutian/nt_transfer/saved_data/supervised_results/'):
    result_dict = {}
    final_result_dict = {}

#     if wiring_str == 'rand':
        # layerwise and global random pruning is the same thing because each parameter will be pruned with an identitical chance.
#         supervised_result_file=supervised_result_file.replace("global", "layerwise")
        
    for density_level in nn_density_list:

        model_path = supervised_result_path + supervised_result_file
        model_wiring_dir = supervised_result_file +  '_' + wiring_str 

        loadFile =  model_path + '/' + model_wiring_dir + '/density_' + str(density_level) + '/supervised_trained_' + supervised_result_file + '_' + 'density_' + str(density_level)

        density_result = list(np.load(loadFile + '.npy' ,allow_pickle='TRUE'))

        density_str = str(density_result[0])

        av_train, trial_variation_train = mean_and_var_across_trials(density_result[1][:, :num_plots] )

        av_test, trial_variation_test = mean_and_var_across_trials(density_result[2][:, :num_plots])

#         av_train, trial_variation_train = mean_and_var_across_trials(density_result[1])

#         av_test, trial_variation_test = mean_and_var_across_trials(density_result[2])

        

        result_density_dict = {'train': {'av': av_train, 'var': trial_variation_train}, 'test': {'av': av_test, 'var': trial_variation_test}}

#         final_result_density_dict = {'train': {'av': av_train[-1], 'var': trial_variation_train[:, -1]}, 'test': {'av': av_test[-1], 'var': trial_variation_test[:, -1]}}
        final_result_density_dict = {'train': {'av': av_train[-1], 'var': trial_variation_train[ -1]}, 'test': {'av': av_test[-1], 'var': trial_variation_test[ -1]}}

        result_dict[str(density_level) ] = result_density_dict

        final_result_dict[str(density_level) ] = final_result_density_dict


    return result_dict, final_result_dict


