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

from .utils import get_tensor, shift_and_scale
from .nets import AffineModule


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
        inputs=1,
        contexts=None,
        bounds=None,
        norm_inputs=None,
        norm_contexts=None,
        transforms=1,
        hidden=1,
        blocks=1,
        activation=torch.relu,
        dropout=0,
        norm_within=False,
        norm_between=False,
        permutation=None,
        linear=None,
        embedding=None,
        distribution=None,
        **kwargs,
        ):
        
        self.inputs = inputs
        self.contexts = contexts
        self.hidden = hidden
        self.blocks = blocks
        self.activation = activation
        self.dropout = dropout
        self.norm_within = norm_within
        
        # Fixed pre-transforms for bounded densities and standardization
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
                        InverseTransform(Exp()),
                        ]))
                    
                else:
                    shift = min(bound)
                    scale = max(bound) - min(bound)
                    featurewise_transform.append(CompositeTransform([
                        InverseTransform(AffineTransform(shift, scale)),
                        InverseTransform(Sigmoid()),
                        ]))
                    
            featurewise_transform = FeaturewiseTransform(featurewise_transform)
            pre_transform.append(featurewise_transform)
            
        if norm_inputs is not None:
            norm_inputs = torch.as_tensor(norm_input)
            assert norm_inputs.size(-1) == inputs
            if bounds is not None:
                norm_inputs = featurewise_transform.forward(norm_inputs)[0]
            norm_transform = AffineTransform(*shift_and_scale(norm_inputs))
            pre_transform.append(norm_transform)
            
        if norm_contexts is not None:
            norm_contexts = torch.as_tensor(norm_contexts)
            assert norm_contexts.size(-1) == contexts
            norm_embedding = AffineModule(*shift_and_scale(norm_contexts))
            if embedding is None:
                embedding = norm_embedding
            else:
                embedding = nn.Sequential(norm_embedding, embedding)
                
        # Main transforms in the flow
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
                    
            main_transform.append(self._get_transform(**kwargs))
            
            if norm_between:
                main_transform.append(BatchNorm(inputs))
                
        transform = CompositeTransform(pre_transform + main_transform)
        if distribution is None:
            distribution = StandardNormal((inputs,))
        super().__init__(transform, distribution, embedding_net=embedding)
        
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
        
        
class MAF(BaseFlow):
    
    def _get_transform(self, residual=False):
        
        return MaskedAffineAutoregressiveTransform(
            self.inputs,
            self.hidden,
            context_features=self.contexts,
            num_blocks=self.blocks,
            use_residual=residual,
            random_mask=False,
            activation=self.activation,
            dropout_probability=dropout,
            use_batch_norm=self.norm_within,
            )
    
    
class CouplingNeuralSplineFlow(BaseFlow):
    
    def _get_transform(self, mask='mid', bins=5, tails='linear', bound=5.0):
        
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
                context_features=self.contexts,
                num_blocks=self.blocks,
                activation=self.activation,
                dropout_probability=self.dropout,
                use_batch_norm=self.norm_within,
                ),
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
            use_batch_norm=self.norm_within,
            )

