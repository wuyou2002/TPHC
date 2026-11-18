from .configuration_llama_mhc import MyLlamaConfig as MHCLlamaConfig
from .modeling_llama_mhc import MyLlamaForCausalLM as MHCLlamaForCausalLM
from .configuration_llama_tphc import MyLlamaConfig as TPHCLlamaConfig
from .modeling_llama_tphc import MyLlamaForCausalLM as TPHCLlamaForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("mhc-llama", MHCLlamaConfig)
AutoModelForCausalLM.register(MHCLlamaConfig, MHCLlamaForCausalLM)
AutoConfig.register("tphc-llama", TPHCLlamaConfig)
AutoModelForCausalLM.register(TPHCLlamaConfig, TPHCLlamaForCausalLM)
