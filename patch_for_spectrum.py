import os
import logging
import sys
import importlib

# Print sys.path for debugging
print(f"DEBUG: sys.path inside patch_for_spectrum.py: {sys.path}")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def patch_worker_init_model(worker_module_name, worker_class_name):
    try:
        spectrum_yaml_path = os.environ.get('SPECTRUM_YAML_PATH')
        apply_freezing = spectrum_yaml_path and os.path.exists(spectrum_yaml_path)

        if apply_freezing:
            logger.info(f"Spectrum patch: Will attempt freezing for {worker_class_name} using {spectrum_yaml_path}")
        else:
            logger.info(f"Spectrum patch: Freezing disabled for {worker_class_name} (SPECTRUM_YAML_PATH not set or file missing)")
            return True # No patching needed if freezing is disabled

        WorkerModule = importlib.import_module(worker_module_name)
        WorkerClass = getattr(WorkerModule, worker_class_name)

        if hasattr(WorkerClass, '_spectrum_patched'):
            logger.warning(f"{worker_class_name}.init_model already patched. Skipping.")
            return True

        if not hasattr(WorkerClass, 'init_model'):
             logger.error(f"Error: {worker_class_name} does not have an init_model method to patch.")
             return False

        orig_init_model = WorkerClass.init_model
        logger.info(f"Patching {worker_class_name}.init_model...")

        def patched_init_model(self, *args, **kwargs):
            logger.info(f"Executing patched init_model for {worker_class_name} instance...")
            orig_init_model(self, *args, **kwargs)
            logger.info(f"Original init_model for {worker_class_name} completed.")

            should_freeze_this_worker = hasattr(self, '_is_actor') and self._is_actor

            if apply_freezing and should_freeze_this_worker:
                target_model = None
                potential_model_attrs = ['actor_module_fsdp', 'actor_module', 'model', '_model']
                for attr_name in potential_model_attrs:
                    if hasattr(self, attr_name):
                        target_model = getattr(self, attr_name)
                        logger.info(f"Found model attribute '{attr_name}' for freezing.")
                        break

                if target_model:
                    logger.info(f"Applying Spectrum parameter freezing to {attr_name}...")
                    try:
                        from verl.utils.spectrum_freezer import apply_spectrum_freezing
                        apply_spectrum_freezing(target_model, spectrum_yaml_path)
                        trainable = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
                        total = sum(p.numel() for p in target_model.parameters())
                        percentage = (trainable / total) * 100 if total > 0 else 0
                        logger.info(f"Spectrum Patch: Freezing applied to {attr_name}. Trainable params: {trainable:,} / {total:,} ({percentage:.2f}%)")
                    except Exception as e_freeze:
                         logger.error(f"Error applying spectrum freezing: {e_freeze}", exc_info=True)
                else:
                    logger.warning(f"Spectrum Patch: Could not find a suitable model attribute on {worker_class_name} instance to apply freezing.")
            elif apply_freezing and not should_freeze_this_worker:
                 logger.info(f"Spectrum Patch: Skipping freezing for this {worker_class_name} instance (not identified as actor).")

        WorkerClass.init_model = patched_init_model
        WorkerClass._spectrum_patched = True
        logger.info(f"Successfully patched {worker_class_name}.init_model")
        return True

    except ImportError as e_import:
        logger.error(f"Actual ImportError: {e_import}", exc_info=True)
        logger.error(f"Error: Could not import {worker_module_name} or find {worker_class_name}.")
        return False
    except Exception as e:
        logger.error(f"Error patching {worker_class_name}: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Applying Spectrum patches...")
    # Only attempt to patch the standard worker now
    success = patch_worker_init_model('verl.workers.fsdp_workers', 'ActorRolloutRefWorker')
    sys.exit(0 if success else 1)
