import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch

from nflows.distributions import StandardNormal
from nflows.flows import Flow
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
from nflows.utils import(
    create_alternating_binary_mask,
    create_mid_split_binary_mask,
    create_random_binary_mask,
    )

from .utils import cpu, device, get_activation, get_optimizer, shift_and_scale
from .nets import NormModule, ForwardNetwork, ResidualNetwork
from .training import Trainer ## TODO


class NormTransform(AffineTransform):

    def __init__(self, inputs):
        
        if type(inputs) is int:
            shift = torch.zeros(inputs)
            scale = torch.ones(inputs)
            
        else:
            shift, scale = shift_and_scale(inputs)
            
        super().__init__(shift, scale)


# Apply indpendent feature-wise (i.e., last axis) transforms
# similar to:
# tnesorflow_probability.bijectors.Blockwise
# torch.distributions.transforms.StackTransform
class FeaturewiseTransform(Transform):

    def __init__(self, transforms):
    
        super().__init__()
        
        self.forwards = [t.forward for t in transforms]
        self.inverses = [t.inverse for t in transforms]
        
    def _map(self, transforms, inputs, context=None):
    
        assert inputs.size(-1) == len(transforms)

        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros(inputs.shape[0])
        
        for i, transform in enumerate(transforms):
            outputs[..., [i]], logabsdet_ = transform(
                inputs[..., [i]], context=context,
                )
            logabsdet += logabsdet_

        return outputs, logabsdet
        
    def forward(self, inputs, context=None):

        return self._map(self.forwards, inputs, context=context)
        
    def inverse(self, inputs, context=None):
        
        return self._map(self.inverses, inputs, context=context)


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
        residual=False, # MLP (False) or residual network (True)
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
        self.residual = residual
        self.blocks = blocks
        self.hidden = hidden
        self.activation = activation
        self.dropout = dropout
        self.batchnorm_within = batchnorm_within
            
        # Zero mean + unit variance per context dimension
        if contexts is not None:
            embedding = self._get_embedding(norm_contexts, embedding)
            
        # Pre-transformations for boundaries and normalization
        pre_transform = self._get_pre_transform(bounds, norm_inputs)
                
        # Main transforms in the flow
        main_transform = self._get_main_transform(
            transforms, permutation, linear, batchnorm_between, kwargs,
            )

        transform = CompositeTransform([pre_transform, main_transform])
        
        if distribution is None:
            distribution = StandardNormal((inputs,))
            
        super().__init__(transform, distribution, embedding_net=embedding)
        
    def prob(self, inputs, context=None):

        return torch.exp(self.log_prob(inputs, context=context))

    # log_prob without scaling factors due to the fixed pre-transforms
    # Based on nflows.distributions.base.Distribution.log_prob
    # and nflows.flows.base.Flow._log_prob
    def _log_prob_without_pre(self, inputs, context=None):
        
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    'Number of inputs must equal number of contexts.'
                    )
                
        context = self._embedding_net(context)
        pre_transform, main_transform = self._transform._transforms
        inputs = pre_transform(inputs, context=context)[0]
        noise, logabsdet = main_transform(inputs, context=context)
        log_prob = self._distribution.log_prob(noise)
        
        return log_prob + logabsdet
    
    def _get_embedding(self, norm_contexts, embedding):
            
        if norm_contexts is not False:
            norm_embedding = NormModule(
                self.contexts if norm_contexts is True else norm_contexts,
                )

            # Rescaling before context embedding network
            if embedding is None:
                embedding = norm_embedding
            else:
                embedding = torch.nn.Sequential(norm_embedding, embedding)
                
        else:
            assert embedding is None
                
        return embedding

    # Combine per-dimension bounds into one bijection
    def _get_bounds_transform(self, bounds):
        
        assert len(bounds) == self.inputs

        # Add bijection required for each dimension
        featurewise_transforms = []
        
        for bound in bounds:

            # Unbounded dimension
            if (bound is None) or all(b is None for b in bound):
                featurewise_transforms.append(IdentityTransform())

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
                featurewise_transforms.append(CompositeTransform([
                    InverseTransform(AffineTransform(shift, scale)),
                    InverseTransform(Exp()),
                    ]))

            # Bounded
            else:
                shift = min(bound)
                scale = max(bound) - min(bound)
                featurewise_transforms.append(CompositeTransform([
                    InverseTransform(AffineTransform(shift, scale)),
                    InverseTransform(Sigmoid()),
                    ]))

        return FeaturewiseTransform(featurewise_transforms)
    
    def _get_norm_transform(self, norm_inputs):
        
        return NormTransform(
            self.inputs if norm_inputs is True else norm_inputs,
            )

    # Fixed pre-transforms for bounded densities and standardization
    def _get_pre_transform(self, bounds, norm_inputs):
        
        pre_transform = IdentityTransform()

        # Enforce boundaries
        if bounds is not None:
            pre_transform = self._get_bounds_transform(bounds)
            
        # Zero mean + unit variance per parameter dimension
        if norm_inputs is not False:
            if norm_inputs is not True:
                norm_inputs = pre_transform(torch.as_tensor(norm_inputs))[0]
            pre_transform = CompositeTransform(
                [pre_transform, self._get_norm_transform(norm_inputs)],
                )
            
        return pre_transform
    
    def _get_main_transform(
        self, transforms, permutation, linear, batchnorm_between, kwargs,
        ):
        
        main_transforms = []
        
        for i in range(transforms):
            
            # Permute parameter order between flow layers
            if permutation is not None:
                if permutation == 'random':
                    main_transforms.append(RandomPermutation(self.inputs))
                elif permutation == 'reverse':
                    main_transforms.append(ReversePermutation(self.inputs))
                else:
                    assert len(permutation) == self.inputs
                    main_transforms.append(Permutation(permutation))
            
            # Linear layer
            if linear is not None:
                if linear == 'lu':
                    main_transforms.append(
                        LULinear(self.inputs, identity_init=True),
                        )
                    
            # Main bijection in this flow layers
            main_transforms.append(self._get_transform(**kwargs))
            
            # Batch normalization at the end of the flow layers
            if batchnorm_between:
                main_transforms.append(BatchNorm(self.inputs))

        return CompositeTransform(main_transforms)
    
    def _get_transform(self, **kwargs):
        
        raise NotImplementedError
        

# Masked affine autoregressive flow
# InverseTransform(AffineAutoregressiveFlow) for inverse autoregressive flow
class AffineAutoregressiveFlow(BaseFlow):
    
    def _get_transform(self, mask=False):
        
        return MaskedAffineAutoregressiveTransform(
            self.inputs,
            self.hidden,
            context_features=self.contexts,
            num_blocks=self.blocks,
            use_residual_blocks=self.residual,
            random_mask=mask,
            activation=get_activation(self.activation, functional=True),
            dropout_probability=self.dropout,
            use_batch_norm=self.batchnorm_within,
            )


class AutoregressiveNeuralSplineFlow(BaseFlow):
    
    def _get_transform(
        self, mask=False, bins=5, tails='linear', bound=5.0,
        ):
        
        return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            self.inputs,
            self.hidden,
            context_features=self.contexts,
            num_bins=bins,
            tails=tails,
            tail_bound=bound,
            num_blocks=self.blocks,
            use_residual_blocks=self.residual,
            random_mask=mask,
            activation=get_activation(self.activation, functional=True),
            dropout_probability=self.dropout,
            use_batch_norm=self.batchnorm_within,
            )


class CouplingNeuralSplineFlow(BaseFlow):
    
    def _get_transform(
        self, mask='mid', bins=5, tails='linear', bound=5.0,
        ):
        
        if type(mask) is str:
            mask = dict(
                alternating=create_alternating_binary_mask(self.inputs),
                mid=create_mid_split_binary_mask(self.inputs),
                random=create_random_binary_mask(self.inputs),
                )[mask]

        net = ResidualNetwork if self.residual else ForwardNetwork
        fn = lambda inputs, outputs: net(
            inputs=inputs,
            outputs=outputs,
            contexts=self.contexts,
            blocks=self.blocks,
            hidden=self.hidden,
            activation=self.activation,
            dropout=self.dropout,
            batchnorm=self.batchnorm_within,
            )

        # if residual:
        #     net = lambda inputs, outputs: ResidualNet(
        #         inputs,
        #         outputs,
        #         context_features=self.contexts,
        #         num_blocks=self.blocks,
        #         hidden_features=self.hidden,
        #         activation=get_activation(self.activation, functional=True),
        #         dropout_probability=self.dropout,
        #         use_batch_norm=self.batchnorm_within,
        #         )
        # else:
        #     net = lambda inputs, outputs: ForwardNet(
        #         inputs=inputs,
        #         outputs=outputs,
        #         contexts=self.contexts,
        #         blocks=self.blocks,
        #         hidden=self.hidden,
        #         activation=self.activation,
        #         dropout=self.dropout,
        #         batchnorm=self.batchnorm_within,
        #         )
        
        return PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=fn,
            num_bins=bins,
            tails=tails,
            tail_bound=bound,
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
    stop_if_inf=True,
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
    best_loss = float('inf')
    losses = {'train': []}
    if validate:
        losses['valid'] = []
    if reduce is not None:
           epoch_reduce = 0

    epoch_loop = range(1, epochs + 1)
    if not verbose:
        epoch_loop = tqdm(epoch_loop)
    for epoch in epoch_loop:
        if verbose:
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
            loop = tqdm(loop, total=n, desc='Train batch', leave=True)
            
        loss_train = 0
        for batch in loop:
            optimizer.zero_grad()
            
            if conditional:
                i, c = batch
                loss_step = loss(i.to(device), c.to(device))
            else:
                loss_step = loss(batch.to(device))
                
            if loss_step.isinf():
                loss_is_inf = True
                if stop_if_inf:
                    break
            else:
                loss_is_inf = False
                loss_step.backward()
                optimizer.step()
                loss_train += loss_step.item()

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
                    loop = tqdm(loop, total=n, desc='Valid batch', leave=True)
                    
                loss_valid = 0
                for batch in loop:
                    
                    if conditional:
                        i, c = batch
                        loss_step = loss(i.to(device), c.to(device))
                    else:
                        loss_step = loss(batch.to(device))
                        
                    if loss_step.isinf():
                        loss_is_inf = True
                        if stop_if_inf:
                            break
                    else:
                        loss_is_inf = False
                        loss_valid += loss_step.item()

                loss_valid /= n
                losses['valid'].append(loss_valid)
                loss_track = loss_valid
                
        if stop_if_inf and loss_is_inf:
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

        if verbose:
            print()
                
    if verbose and save:
        print(save)
        
    model.load_state_dict(best_model)
    model.eval()
                
    return model, losses

