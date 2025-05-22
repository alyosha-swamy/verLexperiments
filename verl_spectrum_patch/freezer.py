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
    
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        # First, freeze all parameters
        logger.info("Freezing all parameters")
        for param in model.parameters():
            param.requires_grad = False
        
        # Then, unfreeze specified parameters
        unfrozen_count = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if self._should_unfreeze(name):
                param.requires_grad = True
                unfrozen_count += 1
                logger.debug(f"Unfreezing parameter: {name}")
        
        unfrozen_percentage = (unfrozen_count / total_params) * 100 if total_params > 0 else 0
        logger.info(f"Unfrozen {unfrozen_count}/{total_params} parameters ({unfrozen_percentage:.2f}%)")
        
        return model
    
    def _should_unfreeze(self, parameter_name: str) -> bool:
        for pattern in self.unfrozen_parameters:
            # Handle regex patterns (typically starting with ^)
            if pattern.startswith('^'):
                if re.match(pattern, parameter_name):
                    return True
            # Handle plain string patterns (direct substring match)
            elif pattern in parameter_name:
                return True
        return False

def apply_spectrum_freezing(model: torch.nn.Module, yaml_path: str) -> torch.nn.Module:
    freezer = SpectrumFreezer(yaml_path)
    return freezer.apply(model) 