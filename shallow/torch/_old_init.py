import torch
from nflows.utils import (
    create_alternating_binary_mask,
    create_mid_split_binary_mask,
    create_random_binary_mask,
    )
from nflows.nn.nets import MLP, ResidualNet
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    InputOutsideDomain,
    Transform,
    CompositeTransform,
    InverseTransform,
    IdentityTransform,
    # AffineTransform,
    Sigmoid,
    BatchNorm,
    Permutation,
    RandomPermutation,
    ReversePermutation,
    LULinear,
    SVDLinear,
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    )


def get_shift_scale(inputs):

    inputs = torch.as_tensor(inputs, dtype=torch.get_default_dtype())
    mean = torch.mean(inputs, dim=0)
    std = inputs.std(dim=0)
    shift = -mean / std
    scale = 1.0 / std

    return shift, scale

    
class AffineModule(torch.nn.Module):
    
    def __init__(self, shift, scale):
        
        super().__init__()
        
        shift = torch.as_tensor(shift, dtype=torch.get_default_dtype())
        scale = torch.as_tensor(scale, dtype=torch.get_default_dtype())    
        self.register_buffer('shift', shift)
        self.register_buffer('scale', scale)
            
    def forward(self, inputs):
        
        return inputs * self.scale + self.shift

    
class AffineTransform(Transform):
    
    def __init__(self, shift, scale):
        
        super().__init__()
        
        shift = torch.as_tensor(shift, dtype=torch.get_default_dtype())
        scale = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        if (scale == 0.0).any():
            raise ValueError('Scale must be non-zero.')
        logabsdet = torch.sum(torch.log(torch.abs(scale)), dim=-1)
    
        self.register_buffer('shift', shift)
        self.register_buffer('scale', scale)
        self.register_buffer('logabsdet', logabsdet)
            
    def forward(self, inputs, context=None):
        
        return inputs * self.scale + self.shift, self.logabsdet
    
    def inverse(self, inputs, context=None):
        
        return (inputs - self.shift) / self.scale, -self.logabsdet


class Exp(Transform):
    
    def forward(self, inputs, context=None):
        
        outputs = torch.exp(inputs)
        logabsdet = torch.sum(inputs, dim=-1)
        
        return outputs, logabsdet
    
    def inverse(self, inputs, context=None):
        
        if torch.min(inputs) <= 0.:
            raise InputOutsideDomain()
            
        outputs = torch.log(inputs)
        logabsdet = -torch.sum(outputs, dim=-1)
        
        return outputs, logabsdet


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

class FeaturewiseTransform_(Transform):
    
    def __init__(self, transforms, axes=None):
        
        super().__init__()
        
        transforms = list(transforms)
        if axes is None:
            axes = [[_] for _ in range(len(transforms))]
        assert len(axes) == len(transforms)
        self._forwards = [transform.forward for transform in transforms]
        self._inverses = [transform.inverse for transform in transforms]
        self.axes = [torch.LongTensor(axis) for axis in axes]
        
    def forward(self, inputs, context=None):
        
        return self._map(self._forwards, inputs, context=context)
    
    def inverse(self, inputs, context=None):
        
        return self._map(self._inverses, inputs, context=context)
    
    def _map(self, transforms, inputs, context=None):
        
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros(list(inputs.shape[:-1])+[len(transforms)])
        for i, (transform, axis) in enumerate(zip(transforms, self.axes)):
            outputs[..., axis], logabsdet[..., i] = transform(
                torch.index_select(inputs, -1, axis), context=context,
                )
        logabsdet = torch.sum(logabsdet, -1)
        
        return outputs, logabsdet
    

# Wrapper inspired by features from sbi and glasflow
# https://github.com/mackelab/sbi/blob/main/sbi/neural_nets/flow.py
# https://github.com/igr-ml/glasflow/blob/main/src/glasflow/flows/coupling.py
class BaseFlow(Flow):

    def __init__(
        self,
        inputs=1,
        conditions=None,
        bounds=None, # None or list of two-item lists
        norm_inputs=None, # inputs to compute mean and std from
        norm_conditions=None, # conditions to compute mean and std from
        transforms=1,
        hidden=1,
        blocks=1, # number of blocks in resnet or layers in mlp
        activation=torch.relu,
        dropout=0.,
        norm_within=False,
        norm_between=False,
        permutation=None, # None, 'reverse', 'random', or list/tuple
        linear=None, # None, 'lu', 'svd'
        embedding=None,
        distribution=None,
        **kwargs,
        ):
        
        self.inputs = inputs
        self.conditions = conditions
        self.hidden = hidden
        self.blocks = blocks
        self.activation = activation
        self.dropout = dropout
        self.norm_within = norm_within
        
        pre_transform = []

        if bounds is not None:
            assert len(bounds) == inputs
            featurewise_transform = []
            for bound in bounds:
                if (bound is None) or all(b is None for b in bound):
                    featurewise_transform.append(IdentityTransform())
                elif any(b is None for b in bound):
                    if bound[0] is None:
                        shift = bound[1]
                        scale = -1.0
                    elif bound[1] is None:
                        shift = bound[0]
                        scale = 1.0
                    featurewise_transform.append(CompositeTransform([
                        InverseTransform(AffineTransform(shift, scale)),
                        InverseTransform(Exp())]))
                else:
                    shift = min(bound)
                    scale = max(bound) - min(bound)
                    featurewise_transform.append(CompositeTransform([
                        InverseTransform(AffineTransform(shift, scale)),
                        InverseTransform(Sigmoid())]))
            featurewise_transform = FeaturewiseTransform(featurewise_transform)
            pre_transform.append(featurewise_transform)                    

        if norm_inputs is not None:
            norm_inputs = torch.as_tensor(norm_inputs)
            assert norm_inputs.size(-1) == inputs
            if bounds is not None:
                norm_inputs = featurewise_transform.forward(norm_inputs)[0]
            norm_transform = AffineTransform(*get_shift_scale(norm_inputs))
            pre_transform.append(norm_transform)
            
        if norm_conditions is not None:
            norm_conditions = torch.as_tensor(norm_conditions)
            assert norm_conditions.size(-1) == conditions
            norm_embedding = AffineModule(*get_shift_scale(norm_conditions))
            if embedding is None:
                embedding = norm_embedding
            else:
                embedding = torch.nn.Sequential(norm_embedding, embedding)
                
        main_transform = []
                
        for i in range(transforms):
            
            if permutation is not None:
                if permutation == 'random':
                    main_transform.append(RandomPermutation(inputs))
                elif permutation == 'reverse':
                    main_transform.append(ReversePermutation(inputs))
                else:
                    main_transform.append(Permutation(permutation))
                    
            if linear is not None:
                if linear == 'lu':
                    main_transform.append(LULinear(inputs))
                elif linear == 'svd':
                    main_transform.append(SVDLinear(inputs, num_householder=10))

            main_transform.append(self._get_transform(**kwargs))
                             
            if norm_between:
                main_transform.append(BatchNorm(inputs))

        transform = CompositeTransform(pre_transform+main_transform)
        if distribution is None:
            distribution = StandardNormal((inputs,))
        super().__init__(transform, distribution, embedding_net=embedding)
        
        self._pre_transform = CompositeTransform(pre_transform)
        self._main_transform = CompositeTransform(main_transform)
            
    def _log_prob_train(self, inputs, context=None):
        
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of inputs must be equal to number of contexts."
                    )

        context = self._embedding_net(context)
        inputs = self._pre_transform(inputs, context=context)[0]
        noise, logabsdet = self._main_transform(inputs, context=context)
        log_prob = self._distribution.log_prob(noise)
        
        return log_prob + logabsdet
        
    def prob(self, inputs, context=None):
        
        return torch.exp(self.log_prob(inputs, context=context))
        
    def _get_transform(self, **kwargs):

        raise NotImplementedError
            
            
class MAF(BaseFlow):
    
    def _get_transform(self, residual=True):
        
        return MaskedAffineAutoregressiveTransform(
            self.inputs,
            self.hidden,
            context_features=self.conditions,
            num_blocks=self.blocks,
            use_residual_blocks=residual,
            random_mask=False,
            activation=self.activation,
            dropout_probability=self.dropout,
            use_batch_norm=self.norm_within,
            )
    
    
class CouplingNSF(BaseFlow):                     

    def _get_transform(self, mask='mid', bins=1, tails='linear', bound=5.):

        return PiecewiseRationalQuadraticCouplingTransform(
            mask=dict(
                alternating=create_alternating_binary_mask(self.inputs),
                mid=create_mid_split_binary_mask(self.inputs),
                random=create_random_binary_mask(self.inputs),
                )[mask] if type(mask) is str else mask,
            transform_net_create_fn=lambda inputs, outputs: ResidualNet(
                inputs,
                outputs,
                hidden_features=self.hidden,
                context_features=self.conditions,
                num_blocks=self.blocks,
                activation=self.activation,
                dropout_probability=self.dropout,
                use_batch_norm=self.norm_within,
                ),
            num_bins=bins,
            tails=tails,
            tail_bound=bound,
            )
    
NSF = CouplingNSF


class AutoregressiveNSF(BaseFlow):
    
    def _get_transform(self, residual=True, mask=False, bins=1, tails='linear', bound=5.0):
        
        return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            self.inputs,
            self.hidden,
            context_features=self.conditions,
            num_bins=bins,
            tails=tails,
            tail_bound=bound,
            num_blocks=self.blocks,
            use_residual_blocks=residual,
            random_mask=mask,
            activation=self.activation,
            dropout_probability=self.dropout,
            use_batch_norm=self.norm_within,
            )

