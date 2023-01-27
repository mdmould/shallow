import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from torch import nn
from nflows.utils import(
    create_alternating_binary_mask,
    create_mid_split_binary_mask,
    create_random_binary_mask,
    )
from nflows.nn.nets import ResidualNet
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import(
    Transform,
    CompositeTransform,
    InverseTransform,
    IdentityTransform,
    PointwiseAffineTransform as AffineTransform,
    Exp,
    Sigmoid,
    BatchNorm,
    Permutation,
    RandomPermutation,
    ReversePermutation,
    LULinear,
    MaskedAffineAutoregressiveTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )

from .utils import cpu, device, get_activation, get_optimizer, shift_and_scale
from .nets import AffineModule
from .train import Trainer


# Apply indpendent feature-wise (i.e., last axis) transforms
# similar to:
# https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Blockwise
# https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#StackTransform
# Details based on https://github.com/bayesiains/nflows/blob/master/nflows/transforms/base.py#L32
class FeaturewiseTransform(Transform):

    def __init__(self, transforms):
    
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)
        self.dim = -1
        
    def _map(self, transforms, inputs, context=None):
    
        assert inputs.size(self.dim) == len(self.transforms)

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs)
        for i, transform in enumerate(transforms):
            outputs[..., [i]], logabsdet[..., i] = transform(
                inputs[..., [i]], context=context)
        logabsdet = torch.sum(logabsdet, dim=self.dim)

        return outputs, logabsdet
        
    def forward(self, inputs, context=None):

        return self._map(
            (t.forward for t in self.transforms), inputs, context=context)
        
    def inverse(self, inputs, context=None):
        
        return self._map(
            (t.inverse for t in self.transforms), inputs, context=context)


# Wrapper inspired by features from sbi and glasflow
# https://github.com/mackelab/sbi/blob/main/sbi/neural_nets/flow.py
# https://github.com/igr-ml/glasflow/blob/main/src/glasflow/flows/coupling.py
# Features include:
# - conditional densities
# - bounded densities
# - standard normalization of inputs and contexts (conditions)
# - embedding network for contexts
# Flows inherit from this base class
# Child classes implement a _get_transform method which take the **kwargs
class BaseFlow(Flow):
    
    def __init__(
        self,
        inputs=1, # Number of parameter dimensions
        contexts=None, # Number of conditional dimensions
        bounds=None, # Parameter boundaries
        norm_inputs=False, # Standardize parameters, bool or array/tensor
        norm_contexts=False, # Standardize contexts, bool or array/tensor
        transforms=1, # Number of flow layers
        blocks=1, # Number of blocks/layers in the net
        hidden=1, # Number of hidden units in each block/layer of the net
        activation='relu', # Activation function
        dropout=0.0, # Dropout probability for hidden units, 0 <= dropout < 1
        batchnorm_within=False, # Batch normalization within the net
        batchnorm_between=False, # Batch normalization between flow layers
        permutation=None, # None, 'random', 'reverse', or list
        linear=None, # None or 'lu'
        embedding=None, # Network to embed contexts
        distribution=None, # None (standard normal) or nflows Distribution
        **kwargs, # Keyword arguments passed to transform constructor
        ):
        
        self.inputs = inputs
        self.contexts = contexts
        self.hidden = hidden
        self.blocks = blocks
        self.activation = get_activation(activation, functional=True)
        self.dropout = dropout
        self.batchnorm_within = batchnorm_within
        
        # Fixed pre-transforms for bounded densities and standardization
        pre_transform = []
        
        # Enforce boundaries
        if bounds is not None:
            assert len(bounds) == inputs
            
            # Add bijection required for each dimension
            featurewise_transform = []
            for bound in bounds:
                
                # Unbounded dimension
                if (bound is None) or all(b is None for b in bound):
                    featurewise_transform.append(IdentityTransform())
                    
                # One side unbounded
                elif any(b is None for b in bound):
                    # Left unbounded
                    if bound[0] is None:
                        shift = bound[1]
                        scale = -1.0
                    # Right unbounded
                    elif bound[1] is None:
                        shift = bound[0]
                        scale = 1.0
                    featurewise_transform.append(CompositeTransform([
                        InverseTransform(AffineTransform(shift, scale)),
                        InverseTransform(Exp()),
                        ]))
                
                # Bounded
                else:
                    shift = min(bound)
                    scale = max(bound) - min(bound)
                    featurewise_transform.append(CompositeTransform([
                        InverseTransform(AffineTransform(shift, scale)),
                        InverseTransform(Sigmoid()),
                        ]))
                    
            # Combine per-dimension bijections into one bijection
            featurewise_transform = FeaturewiseTransform(featurewise_transform)
            pre_transform.append(featurewise_transform)
            
        # Zero mean + unit variance per parameter dimension
        if norm_inputs is not False:
            # Place holder for loading state dict
            if norm_inputs is True:
                shift, scale = torch.zeros(inputs), torch.ones(inputs)
            # Input tensor to compute mean and variance from
            else:
                norm_inputs = torch.as_tensor(norm_inputs)
                assert norm_inputs.size(-1) == inputs
                # Rescaling after boundary-enforcing bijection
                if bounds is not None:
                    norm_inputs = featurewise_transform.forward(norm_inputs)[0]
                shift, scale = shift_and_scale(norm_inputs)
            norm_transform = AffineTransform(shift, scale)
            pre_transform.append(norm_transform)
            
        if contexts is not None:
            # Zero mean + unit variance per context dimension
            if norm_contexts is not False:
                # Place holder for loading state dict
                if norm_contexts is True:
                    shift, scale = torch.zeros(contexts), torch.ones(contexts)
                # Input tensor to compute mean and variance from
                else:
                    norm_contexts = torch.as_tensor(norm_contexts)
                    assert norm_contexts.size(-1) == contexts
                    shift, scale = shift_and_scale(norm_contexts)
                norm_embedding = AffineModule(shift, scale)
                # Rescaling before context embedding network
                if embedding is None:
                    embedding = norm_embedding
                else:
                    embedding = nn.Sequential(norm_embedding, embedding)
        else:
            assert norm_contexts is False and embedding is None
                
        # Main transforms in the flow
        main_transform = []
        
        for i in range(transforms):
            
            # Permute parameter order between flow layers
            if permutation is not None:
                if permutation == 'random':
                    main_transform.append(RandomPermutation(inputs))
                elif permutation == 'reverse':
                    main_transform.append(ReversePermutation(inputs))
                else:
                    main_transform.append(Permutation(permutation))
            
            # Linear layer
            if linear is not None:
                if linear == 'lu':
                    main_transform.append(LULinear(inputs, identity_init=True))
                    
            # Main bijection in this flow layers
            main_transform.append(self._get_transform(**kwargs))
            
            # Batch normalization at the end of the flow layers
            if batchnorm_between:
                main_transform.append(BatchNorm(inputs))
                
        transform = CompositeTransform(pre_transform + main_transform)
        if distribution is None:
            distribution = StandardNormal((inputs,))
        super().__init__(transform, distribution, embedding_net=embedding)
        
        ## TODO: don't double register modules/parameters
        self._pre_transform = CompositeTransform(pre_transform)
        self._main_transform = CompositeTransform(main_transform)
        
    def prob(self, inputs, context=None):
        
        return torch.exp(self.log_prob(inputs, context=context))
    
    # log_prob without scaling factors due to the fixed pre-transforms
    # Based on https://github.com/bayesiains/nflows/blob/master/nflows/distributions/base.py#L16
    def _log_prob_without_pre(self, inputs, context=None):
        
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    'Number of inputs must equal number of contexts.'
                    )
                
        context = self._embedding_net(context)
        inputs = self._pre_transform(inputs, context=context)[0]
        noise, logabsdet = self._main_transform(inputs, context=context)
        log_prob = self._distribution.log_prob(noise)
        
        return log_prob + logabsdet
    
    def _get_transform(self, **kwargs):
        
        raise NotImplementedError
        

# Masked affine autoregressivle flow
# Use InverseTransform(AffineAutoregressiveFlow)
# for inverse autoregressive flow
class AffineAutoregressiveFlow(BaseFlow):
    
    def _get_transform(self, residual=False, mask=False):
        
        return MaskedAffineAutoregressiveTransform(
            self.inputs,
            self.hidden,
            context_features=self.contexts,
            num_blocks=self.blocks,
            use_residual_blocks=residual,
            random_mask=mask,
            activation=self.activation,
            dropout_probability=self.dropout,
            use_batch_norm=self.batchnorm_within,
            )
    

## TODO: allow non-residual blocks
class CouplingNeuralSplineFlow(BaseFlow):
    
    def _get_transform(
        self, residual=True, mask='mid', bins=5, tails='linear', bound=5.0,
        ):
        
        if type(mask) is str:
            mask = dict(
                alternating=create_alternating_binary_mask(self.inputs),
                mid=create_mid_split_binary_mask(self.inputs),
                random=create_random_binary_mask(self.inputs),
                )[mask]
            
        if residual:
            net = lambda inputs, outputs: ResidualNet(
                inputs,
                outputs,
                hidden_features=self.hidden,
                context_features=self.contexts,
                num_blocks=self.blocks,
                activation=self.activation,
                dropout_probability=self.dropout,
                use_batch_norm=self.batchnorm_within,
                )
        else:
            net = lambda inputs, outputs: None
        
        return PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=net,
            num_bins=bins,
            tails=tails,
            tail_bound=bound,
            )


class AutoregressiveNeuralSplineFlow(BaseFlow):
    
    def _get_transform(
        self, residual=False, mask=False, bins=5, tails='linear', bound=5.0,
        ):
        
        return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            self.inputs,
            self.hidden,
            context_features=self.contexts,
            num_bins=bins,
            tails=tails,
            tail_bound=bound,
            num_blocks=self.blocks,
            use_residual_blocks=residual,
            random_mask=mask,
            activation=self.activation,
            dropout_probability=self.dropout,
            use_batch_norm=self.batchnorm_within,
            )
    

## TODO: sub-class train.Trainer
def trainer(
    model,
    inputs,
    contexts=None,
    inputs_valid=None,
    contexts_valid=None,
    loss=None,
    optimizer='adam',
    learning_rate=1e-3,
    weight_decay=0,
    epochs=1,
    batch_size=None,
    batch_size_valid='train',
    shuffle=True,
    reduce=None,
    stop=None,
    stop_ifnot_finite=True,
    verbose=True,
    save=None,
    seed=None,
    ):
    
    if seed is not None:
        torch.manual_seed(seed)
        
    model.to(device)
    
    inputs = torch.as_tensor(inputs, dtype=torch.float32, device=cpu)
    if inputs.ndim == 1:
        inputs = inputs[..., None]
        
    conditional = False
    if contexts is not None:
        conditional = True
        
        contexts = torch.as_tensor(contexts, dtype=torch.float32, device=cpu)
        if contexts.ndim == 1:
            contexts = contexts[..., None]
        assert contexts.shape[0] == inputs.shape[0]
        
    validate = False
    if inputs_valid is not None:
        validate = True
        
        inputs_valid = torch.as_tensor(
            inputs_valid, dtype=torch.float32, device=cpu,
            )
        if inputs_valid.ndim == 1:
            inputs_valid = inputs_valid[..., None]
        assert inputs_valid.shape[-1] == inputs.shape[-1]
                        
        if conditional:
            assert contexts_valid is not None
            
            contexts_valid = torch.as_tensor(
                contexts_valid, dtype=torch.float32, device=cpu,
                )
            if contexts_valid.ndim == 1:
                contexts_valid = contexts_valid[..., None]
            assert contexts_valid.shape[0] == inputs_valid.shape[0]
            assert contexts_valid.shape[-1] == contexts.shape[-1]
            
            if (batch_size_valid is None or
                ((batch_size_valid == 'train') and (batch_size is None))
                ):
                contexts_valid = contexts_valid[None, ...]
            else:
                if batch_size_valid == 'train':
                    batch_size_valid = batch_size
                contexts_valid = contexts_valid.split(batch_size_valid)
                
        if (batch_size_valid is None or
            ((batch_size_valid == 'train') and (batch_size is None))
            ):
            inputs_valid = inputs_valid[None, ...]
        else:
            if batch_size_valid == 'train':
                batch_size_valid = batch_size
            inputs_valid = inputs_valid.split(batch_size_valid)
                
    if not shuffle:
        if batch_size is None:
            inputs = inputs[None, ...]
        else:
            inputs = inputs.split(batch_size)
            
        if conditional:
            if batch_size is None:
                contexts = contexts[None, ...]
            else:
                contexts = contexts.split(batch_size)
                
    if loss is None:
        loss = lambda i, c=None: -model.log_prob(i, context=c).mean()
    assert callable(loss)
    
    optimizer = get_optimizer(optimizer)(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    
    best_model = deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = np.inf
    losses = {'train': []}
    if validate:
        losses['valid'] = []
    if reduce is not None:
           epoch_reduce = 0
        
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}')
        
        # Training
        model = model.train()
        
        if shuffle:
            perm = torch.randperm(inputs.shape[0])
            
            inputs_train = inputs[perm]
            if batch_size is None:
                inputs_train = inputs_train[None, ...]
            else:
                inputs_train = inputs_train.split(batch_size)
                
            if conditional:
                contexts_train = contexts[perm]
                if batch_size is None:
                    contexts_train = contexts_train[None, ...]
                else:
                    contexts_train = contexts_train.split(batch_size)
            
        else:
            inputs_train = inputs
            if conditional:
                contexts_train = contexts
                
        n = len(inputs_train)
        if conditional:
            loop = zip(inputs_train, contexts_train)
        else:
            loop = inputs_train
        if verbose:
            loop = tqdm(loop, total=n)
            
        loss_train = 0
        for batch in loop:
            optimizer.zero_grad()
            
            if conditional:
                i, c = batch
                loss_step = loss(i.to(device), c.to(device))
            else:
                loss_step = loss(batch.to(device))
                
            if loss_step.isfinite():
                loss_isfinite = True
                loss_step.backward()
                optimizer.step()
                loss_train += loss_step.item()
            else:
                loss_isfinite = False
                if stop_ifnot_finite:
                    break

        loss_train /= n
        losses['train'].append(loss_train)
        loss_track = loss_train
        
        # Validation
        if validate:
            model = model.eval()
            with torch.inference_mode():
                
                n = len(inputs_valid)
                if conditional:
                    loop = zip(inputs_valid, contexts_valid)
                else:
                    loop = inputs_valid
                if verbose:
                    loop = tqdm(loop, total=n)
                    
                loss_valid = 0
                for batch in loop:
                    
                    if conditional:
                        i, c = batch
                        loss_step = loss(i.to(device), c.to(device))
                    else:
                        loss_step = loss(batch.to(device))
                        
                    if loss_step.isfinite():
                        loss_isfinite = True
                        loss_valid += loss_step.item()
                    else:
                        loss_isfinite = False
                        if stop_ifnot_finite:
                            break

                loss_valid /= n
                losses['valid'].append(loss_valid)
                loss_track = loss_valid
                
        if stop_ifnot_finite and not loss_isfinite:
            print('nan/inf loss, stopping')
            break
            
        if verbose:
            print(loss_train, end='')
            if validate:
                print(f', {loss_valid}', end='')
            print()
            
        if save is not None:
            np.save(f'{save}.npy', losses, allow_pickle=True)
            
        if loss_track < best_loss:
            if verbose:
                print('Loss improved')
            best_epoch = epoch
            best_loss = loss_track
            best_model = deepcopy(model.state_dict())
            if save is not None:
                torch.save(best_model, f'{save}.pt')
                
        if reduce is not None:
            if epoch - best_epoch == 0:
                epoch_reduce = epoch
            if epoch - epoch_reduce > reduce:
                epoch_reduce = epoch
                if verbose:
                    print(f'No improvement for {reduce} epochs, reducing lr')
                for group in optimizer.param_groups:
                    group['lr'] /= 2
                    
        if stop is not None:
            if epoch - best_epoch > stop:
                if verbose:
                    print(f'No improvement for {stop} epochs, stopping')
                break
                
    if verbose and save:
        print(save)
        
    model.load_state_dict(best_model)
    model.eval()
                
    return model, losses

