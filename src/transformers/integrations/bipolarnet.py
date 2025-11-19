# This implementation is adapted from the BitNet integration in Hugging Face Transformers
# (https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/bitnet.py).
# The code has been modified to support a 0/1 binary quantization scheme with separate positive and negative weight branches, custom scaling logic, and updated loading/unpacking behavior.

from ..utils import is_accelerate_available, is_torch_available, logging


if is_accelerate_available():
    from accelerate import init_empty_weights

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

logger = logging.get_logger(__name__)


VALUES_PER_ITEM = 8


def pack_weights(quantized_weights: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of binary weights (-1/+1) into a compact 1-bit-per-value format.
    Each uint8 stores 8 binary values.

    Conversion:
        -1 -> 0
        +1 -> 1

    Parameters
    ----------
    quantized_weights : torch.Tensor
        Tensor with values in {-1, +1}, shape: [out_features, in_features] or [in_features]

    Returns
    -------
    torch.Tensor
        Packed tensor storing 8 binary values per uint8.
        Shape: [out_features, ceil(in_features/8)] or [ceil(in_features/8)]
    """
    
    # Convert -1/+1 to 0/1
    quantized_weights = ((quantized_weights + 1) // 2).to(torch.uint8)

    quantized_weights = quantized_weights.transpose(0, -1).contiguous()
    original_shape = quantized_weights.shape

    row_dim = (original_shape[0] + VALUES_PER_ITEM - 1) // VALUES_PER_ITEM

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    packed = torch.zeros(packed_tensor_shape, device=quantized_weights.device, dtype=torch.uint8)
    unpacked = quantized_weights.to(torch.uint8)

    for i in range(row_dim):
        for bit in range(VALUES_PER_ITEM):
            idx = i * VALUES_PER_ITEM + bit
            if idx < original_shape[0]:
                packed[i] |= (unpacked[idx] & 1) << bit
                
    packed = packed.transpose(0, -1).contiguous()
    return packed


@torch.compile
def unpack_weights(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Unpacks bipolar binary weights that were stored using 1 bit per value.
    Converts the unpacked bits from {0,1} back to {-1,+1}.

    Parameters:
    -----------
    packed : torch.Tensor
        Packed tensor with 8 weights per uint8.
    dtype : torch.dtype
        Desired output dtype.

    Returns:
    --------
    torch.Tensor
        Unpacked bipolar weights in {-1, +1} and cast to dtype.
    """
    
    packed = packed.transpose(0, -1).contiguous()
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    for i in range(packed_shape[0]):
        for bit in range(VALUES_PER_ITEM):
            idx = i * VALUES_PER_ITEM + bit
            unpacked[idx] = (packed[i] >> bit) & 1
    
    # Convert 0/1 back to -1/+1
    unpacked = unpacked * 2 - 1
    unpacked = unpacked.transpose(0, -1).contiguous()
    return unpacked.to(dtype)


class BipolarLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device=None,
        dtype=None,
        use_rms_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer(
            "weight_pos",
            torch.zeros((out_features, in_features // VALUES_PER_ITEM), dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "weight_neg",
            torch.zeros((out_features, in_features // VALUES_PER_ITEM), dtype=torch.uint8, device=device),
        )
        
        self.register_buffer(
            "weight_scale_pos",
            torch.ones((out_features,), dtype=dtype, device=device)
        )
        self.register_buffer(
            "weight_scale_neg",
            torch.ones((out_features,), dtype=dtype, device=device)
        )
            
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype, device=device))
        else:
            self.bias = None

        # Optional RMSNorm (applied on the activations before quantization).
        self.rms_norm = None
        if use_rms_norm:
            from ..models.llama.modeling_llama import LlamaRMSNorm
            self.rms_norm = LlamaRMSNorm(in_features, eps=rms_norm_eps)

    @torch.compile
    def activation_quant(self, input, num_bits=8):
        """
        Activation function : Performs symmetric, per-token quantization on the input activations.
        Parameters:
        -----------
        x : torch.Tensor [batch, seq, hidden]
            Input activations to be quantized.
        num_bits : int, optional (default=8)
            Number of bits to use for quantization, determining the quantization range.

        Returns:
        --------
        result : torch.Tensor [batch, seq, hidden]
            Quantized activation tensor, with values mapped to an `int8` range.
        scale : torch.Tensor [batch, seq, 1]
            The per-channel scaling factors used to quantize the tensor.
        """
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1
        scale = Qp / input.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (input * scale).round().clamp(Qn, Qp)
        return result.to(torch.int8), scale

    def forward(self, input):
        # Apply RMSNorm on the input if requested.
        if self.rms_norm is not None:
            input = self.rms_norm(input)
        w_pos = unpack_weights(self.weight_pos, dtype=self.dtype)
        w_neg = unpack_weights(self.weight_neg, dtype=self.dtype)
        input_quant, input_scale = self.activation_quant(input)
        
        y_pos = F.linear(input_quant.to(self.dtype), w_pos)
        y_neg = F.linear(input_quant.to(self.dtype), w_neg)
        
        y = (y_pos / self.weight_scale_pos) - (y_neg / self.weight_scale_neg)
        y = y / input_scale
        
        if self.bias is not None:
            y += self.bias.view(1, -1).expand_as(y)
        return y


class WeightQuant(torch.autograd.Function):
    """
    Implements a custom autograd function for weight quantization.
    This performs binary quantization (-1, 1) based on scaling by the
    mean value of the positive weights. It uses the Straight-Through Estimator
    (STE) for the backward pass.
    """

    @staticmethod
    def forward(ctx, weight):
        dtype = weight.dtype
        weight = weight.float()
        scale = 1.0 / weight.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
        weight = weight.sign() / scale
        return weight.to(dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ActQuant(torch.autograd.Function):
    """
    Implements a custom autograd function for activation quantization.
    This performs symmetric 8-bit quantization (to the range [-128, 127])
    based on the maximum absolute value along the last dimension (per-token/row scaling).
    It uses the Straight-Through Estimator (STE) for the backward pass.
    """

    @staticmethod
    def forward(ctx, activation):
        dtype = activation.dtype
        activation = activation.float()
        scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        activation = (activation * scale).round().clamp(-128, 127) / scale
        return activation.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class AutoBipolarLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        online_quant: bool = False,
        use_rms_norm: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.online_quant = online_quant
        self.dtype = dtype
        
        # Optional RMSNorm
        self.rms_norm = None
        if use_rms_norm:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            self.rms_norm = LlamaRMSNorm(in_features, eps=rms_norm_eps)
        
        self.weight_pos = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        self.weight_neg = nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.bias = None

        if not online_quant:
            self.register_buffer("weight_scale_pos", torch.ones(out_features, dtype=dtype, device=device))
            self.register_buffer("weight_scale_neg", torch.ones(out_features, dtype=dtype, device=device))

            self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict,
        prefix,
        *args,
        **kwargs,
    ):
        for name in ["weight_pos", "weight_neg"]:
            key = f"{prefix}{name}.weight"
            if key in state_dict:
                module = getattr(self, name)
                if state_dict[key].dtype != module.weight.dtype:
                    state_dict[key] = unpack_weights(
                        state_dict[key], dtype=module.weight.dtype
                    )
        return state_dict

    def forward(self, input):
        # Optional RMSNorm on activations prior to quantization.
        if self.rms_norm is not None:
            input = self.rms_norm(input)

        if self.online_quant:
            weight_pos = WeightQuant.apply(self.weight_pos.weight)
            weight_neg = WeightQuant.apply(self.weight_neg.weight)
        else:
            weight_pos = self.weight_pos.weight / self.weight_scale_pos[:, None]
            weight_neg = self.weight_neg.weight / self.weight_scale_neg[:, None]
        
        input = ActQuant.apply(input)
        output = F.linear(input, weight_pos - weight_neg, self.bias)
        return output

    
def _replace_with_bipolarnet_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """

    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        # Check if the current key is not in the `modules_to_not_convert`
        if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
            with init_empty_weights():
                if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                    in_features = module.in_features
                    out_features = module.out_features
                    if quantization_config and quantization_config.linear_class == "autobipolarlinear":
                        model._modules[name] = AutoBipolarLinear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=module.bias is not None,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            online_quant=(quantization_config.quantization_mode == "online"),
                            use_rms_norm=quantization_config.use_rms_norm,
                            rms_norm_eps=quantization_config.rms_norm_eps,
                        )
                        if quantization_config.quantization_mode == "offline":
                            model._modules[name].requires_grad_(False)
                    else:
                        model._modules[name] = BipolarLinear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=module.bias is not None,
                            device=module.weight.device,
                            dtype=module.weight.dtype,
                            use_rms_norm=quantization_config.use_rms_norm if quantization_config else False,
                            rms_norm_eps=quantization_config.rms_norm_eps if quantization_config else 1e-6,
                        )
                        model._modules[name].requires_grad_(False)
                    has_been_replaced = True

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bipolarnet_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bipolarnet_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    pre_quantized=False,
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `BipolarLinear` modules`.

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Each weight will be quantized along the channel.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `BipolarLinear`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`list[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    if quantization_config and quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_bipolarnet_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        pre_quantized=pre_quantized,
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using bipolarnet but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
