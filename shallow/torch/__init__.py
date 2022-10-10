import torch
from nflows.utils import (
    sum_except_batch,
    tensor2numpy,
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
    AffineTransform,
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


class Affine(torch.nn.Module):
    
    def __init__(self, shift, scale):
        
        super().__init__()
        
        self.shift = shift
        self.scale = scale
        
    def forward(self, inputs):
        
        return inputs * scale + shift


class ExpTransform(Transform):
    
    def forward(self, inputs, context=None):
        
        outputs = torch.exp(inputs)
        logabsdet = torch.log(torch.sum(outputs, dim=-1))
        
        return outputs, logabsdet
    
    def inverse(self, inputs, context=None):
        
        if torch.min(inputs) <= 0.:
            raise InputOutsideDomain()
            
        outputs = torch.log(inputs)
        logabsdet = -torch.sum(outputs, dim=-1)
        
        return outputs, logabsdet
    

# Apply component-wise transforms
# https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/Blockwise
# https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#CatTransform
class BlockwiseTransform(Transform):
    
    def __init__(self, transforms, axes=None):
        
        super().__init__()
        
        transforms = list(transforms)
        if axes is None:
            axes = [[_] for _ in range(len(transforms))]
        assert len(axes) == len(self.transforms)
        self._forwards = [transform.forward for transform in transforms]
        self._inverses = [transform.inverse for transform in transforms]
        self.axes = [torch.LongTensor(axis) for axis in axes]
        self.dim = dim
        
    def forward(self, inputs, context=None):
        
        return self._map(self._forwards, inputs, context=context)
    
    def inverse(self, inputs, context=None):
        
        return self._map(self._inverses, inputs, context=context)
    
    def _map(self, transforms, inputs, context=None):
        
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros(list(inputs.shape[:-1])+[len(self.transforms)])
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
        hidden=1,
        blocks=1, # number of blocks in resnet or layers in mlp
        activation=torch.relu,
        dropout=0.,
        norm_within=False,
        transforms=1,
        permutation='reverse', # 'reverse', 'random', or list/tuple
        linear=None, # None, 'lu', 'svd'
        norm_inputs=None, # inputs to compute mean and std from
        norm_conditions=None, # conditions to compute mean and std from
        norm_between=False,
        emedding=torch.nn.Identity(),
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
        
        transform = []
        
        if bounds is not None:
            blockwise_transform = self._get_blockwise(bounds)
            transform.append(blockwise_transform)
            
        if norm_inputs is not None:
            if bounds is not None:
                norm_inputs = blockwise_transform.forward(norm_inputs)
            transform.append(self._get_norm(norm_inputs))
        
        embedding = self._get_embedding(norm_conditions, embedding)
            
        for i in range(transforms):
            
            if permutation is not None:
                transform.append(self._get_permutation(inputs, permutation))
                
            if linear is not None:
                transform.append(self._get_linear(inputs, linear))
                
            transform.append(self._get_transform(**kwargs))
                             
            if norm_between:
                transform.append(BatchNorm(inputs))
                             
        transform = CompositeTransform(transform)
        
        if distriubtion is None:
            distribution = StandardNormal((inputs,))
        
        super().__init__(transform, distribution, embedding_net=embedding)
        
        def _get_transform(self, **kwargs):
            
            return None
                             
        def _get_blockwise(self, bounds):
            
            transform = []
            axes = None
            for bound in bounds:
                if (bound is None) or all(b is None for b in bound):
                    transform.append(IdentityTransform())
                elif any(b is None for b in bound):
                    if b[0] is None:
                        shift = b[1]
                        scale = -1.
                    elif b[1] is None:
                        shift = b[1]
                        scale = 1.
                    transform.append(CompositeTransform([
                        InverseTransform(AffineTransform(shift, scale)),
                        InverseTransform(ExpTransform()),
                        ]))
                else:
                    shift = min(bound)
                    scale = max(bound) - min(bound)
                    transform.append(CompositeTransform([
                        InverseTransform(AffineTransform(shift, scale)),
                        InverseTransform(Sigmoid()),
                        ]))
            
            return BlockwiseTransform(transform, axes=axes)
        
        def _get_norm(self, norm_inputs):
        
            mean = torch.mean(norm_inputs, dim=0)
            std = torch.std(norm_inputs, dim=0)
            shift = -mean / std
            scale = 1. / std

            return AffineTransform(shift, scale)
                             
        def _get_embedding(self, norm_conditions, embedding):
            
            if norm_conditions is not None:
                mean = torch.mean(norm_conditions, dim=0)
                std = torch.std(norm_conditions, dim=0)
                shift = -mean / std
                scale = 1. / std
                embedding = nn.Sequntial(Affine(shift, scale), embedding)
                
            return embedding
            
        def _get_permutation(self, inputs, permutation):
            
            if permuation == 'random':
                return RandomPermutation(inputs)
            elif permutation == 'reverse':
                return ReversePermutation(inputs)
            else:
                return Permutation(permutation)
            
        def _get_linear(self, inputs, linear):
            
            if linear == 'lu':
                return LULinear(inputs, using_cache=False, identity_init=True)
            elif linear == 'svd':
                return SVDLinear(
                    inputs, num_householder=10, using_cache=False,
                    identity_init=True,
                    )
            
            
def MAF(BaseFlow):
    
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
    
    
class NSF(BaseFlow):                     

    def _get_transform(
        self,
        bins=1,
        tails='linear',
        bound=3.,
        mask='mid',
        ):

        return PiecewiseRationalQuadraticCouplingTransform(
            mask=self._get_mask(mask),
            transform_net_create_fn=self._get_net(),
            num_bins=bins,
            tails=tails,
            tail_bound=bound,
            )

    def _get_mask(self, mask):

        if mask == 'alternating':
            return create_alternating_binary_mask(self.inputs, even=True)
        elif mask == 'mid':
            return create_mid_split_binary_mask(self.inputs)
        elif mask == 'random':
            return create_random_binary_mask(self.inputs)
        else:
            return mask

    def _get_net(self):

        return lambda ins, outs: ResidualNet(
            ins,
            out,
            hidden_features=self.hidden,
            context_features=self.conditions,
            num_blocks=self.blocks,
            activation=aself.ctivation,
            dropout_probability=self.dropout,
            use_batch_norm=self.norm_within,
            )

