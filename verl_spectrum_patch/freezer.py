import yaml
import torch
import logging
import re
from typing import List

logger = logging.getLogger(__name__)

class SpectrumFreezer:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.unfrozen_parameters = self._load_unfrozen_parameters()
    
    def _load_unfrozen_parameters(self) -> List[str]:
        try:
            with open(self.yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            if not config or 'unfrozen_parameters' not in config:
                logger.warning(f"No 'unfrozen_parameters' found in {self.yaml_path}")
                return []
            return config['unfrozen_parameters']
        except Exception as e:
            logger.error(f"Error loading YAML file {self.yaml_path}: {e}")
            return []

    def apply_freezing(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.unfrozen_parameters:
            logger.warning("No unfrozen_parameters specified in YAML or YAML load failed. No parameters will be unfrozen by SpectrumFreezer (all trainable).")
            # Optionally, freeze all parameters if none are specified to be unfrozen
            # for param in model.parameters():
            #     param.requires_grad = False
            # logger.warning("Since no unfrozen_parameters were specified, ALL parameters have been frozen.")
            return model

        logger.info(f"SpectrumFreezer: Attempting to unfreeze parameters matching: {self.unfrozen_parameters}")
        
        total_params = 0
        trainable_params = 0
        
        # First, freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze only the specified ones
        unfrozen_found_any = False
        for name, param in model.named_parameters():
            total_params += param.numel()
            param_is_unfrozen = False
            for unfrozen_pattern in self.unfrozen_parameters:
                if re.search(unfrozen_pattern, name):
                    param.requires_grad = True
                    param_is_unfrozen = True
                    unfrozen_found_any = True
                    logger.info(f"  Unfreezing '{name}' due to pattern '{unfrozen_pattern}'")
                    break # Match found, no need to check other patterns for this param
            if param.requires_grad:
                trainable_params += param.numel()
        
        if not unfrozen_found_any:
            logger.warning("SpectrumFreezer: No parameters were unfrozen. Check your YAML patterns against model parameter names.")
        
        logger.info(f"SpectrumFreezer: Model freezing applied. Trainable parameters: {trainable_params}/{total_params} ({100 * trainable_params / total_params:.2f}%)")
        return model

def apply_spectrum_freezing(model: torch.nn.Module, yaml_path: str) -> torch.nn.Module:
    freezer = SpectrumFreezer(yaml_path)
    return freezer.apply_freezing(model) 