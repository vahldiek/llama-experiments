import time
import importlib
import logging
import torch
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel
from typing import List, Union
import intel_extension_for_pytorch as ipex



logger = logging.getLogger('ipex_inference_transformers')


#Creating a template class for initializing different types of transfomers
class IpexAutoInferenceTransformer:
    #For now only support Llama.  Should work for others but only have infrastructure to
    #test Llama
    #First element of tuple is class from huggingface .config
    #Second element of tuple is the class to map to
    supported_models = [("LlamaForCausalLM", "LlamaForCausalLM"), ("LlamaModel", "LlamaForCausalLM")]
    supported_model_architectures = [x[0] for x in supported_models]
    #override all types of init.  This class should only be used statically
    def __init__(self, *args, **kwargs):
        logger.error("Cannot create instance of IpexAutoInferenceTransformer.  Use from_ipex_pretrained static method")
        raise NotImplementedError()
    
    #Do not use this method, use from_ipex_pretrained
    @classmethod
    def from_ipex_pretrained(cls, base_model_name_or_path: str, quantized_model_path: str, config=None) -> PreTrainedModel:
        if config is None:
            config = AutoConfig.from_pretrained(base_model_name_or_path, torchscript=True)

        architectures = config.architectures
        #Walk through the architectures and see if we find one that we can support
        for architecture in architectures:
            model_index = IpexAutoInferenceTransformer.supported_model_architectures.index(architecture)
            if model_index >= 0:
                transformers_module = importlib.import_module("transformers")
                base_model_cls = getattr(transformers_module, IpexAutoInferenceTransformer.supported_models[model_index][1])
                
                #begin optimization for using Intel quantized model
                torch._C._jit_set_texpr_fuser_enabled(False)
                qconfig = ipex.quantization.default_static_qconfig_mapping
                #Just load a shell of the base model and replace it with ipex model
                num_hidden_layers = config.num_hidden_layers
                hidden_size = config.hidden_size
                config.num_hidden_layers = 0
                config.hidden_size = 0
                base_model = base_model_cls(config=config)
                base_model.config.num_hidden_layers = num_hidden_layers
                base_model.config.hidden_size = hidden_size
                logger.debug(f"Created {architecture} model object")

                #"monkey patches" the original_model object to swap in a few optimized functions
                base_model = ipex.optimize_transformers(
                    base_model.eval(),
                    dtype=torch.float,
                    inplace=True,
                    quantization_config=qconfig,
                    deployment_mode=False)

                logger.debug("About to load quantized model")
                #Load the Intel quantized model
                start = time.perf_counter()
                self_jit = torch.jit.load(quantized_model_path)
                end = time.perf_counter()
                self_jit = torch.jit.freeze(self_jit.eval())
                logger.debug(f"quantized model load took {end - start:0.4f} seconds")
                #Set self_jit as the optimized model
                ipex._set_optimized_model_for_generation(base_model, optimized_model=self_jit)
                return base_model
            

        logger.warn(f"Cannot find supported class for {base_model_name_or_path}")
        logger.warn(f"Available classes were {architectures}")
        logger.warn(f"Supported classes are {cls.supported_model_architectures}")
        return None

            
        



    