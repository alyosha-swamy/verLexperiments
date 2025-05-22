import os
import sys
import logging
from importlib import import_module
import datetime

# Create a proof file that this script has run
# This helps debug if stdout/stderr is not captured from workers.
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
pid = os.getpid()
proof_file_path = f"/tmp/sitecustomize_executed_pid_{pid}_{timestamp}.txt"

def _log_to_proof_file(message):
    with open(proof_file_path, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} - {message}\n")

_log_to_proof_file(f"DEBUG_SITECUSTOMIZE: sitecustomize.py in verl_spectrum_patch is attempting to run. PID: {pid}")
_log_to_proof_file(f"DEBUG_SITECUSTOMIZE: PYTHONPATH: {os.getenv('PYTHONPATH')}")
_log_to_proof_file(f"DEBUG_SITECUSTOMIZE: sys.path: {sys.path}")

print(f"DEBUG_SITECUSTOMIZE: sitecustomize.py in verl_spectrum_patch is attempting to run. PID: {pid} PYTHONPATH: {os.getenv('PYTHONPATH')}")

log = logging.getLogger(__name__) # Keep this logger if verl uses it, otherwise can remove
YAML = os.getenv("SPECTRUM_YAML_PATH")

_log_to_proof_file(f"DEBUG_SITECUSTOMIZE: SPECTRUM_YAML_PATH from env: {YAML}")
print(f"DEBUG_SITECUSTOMIZE: SPECTRUM_YAML_PATH from env: {YAML}")

if YAML and os.path.exists(YAML):
    _log_to_proof_file(f"DEBUG_SITECUSTOMIZE: YAML path {YAML} exists. Attempting to patch.")
    print(f"DEBUG_SITECUSTOMIZE: YAML path {YAML} exists. Attempting to patch.")
    try:
        _log_to_proof_file("DEBUG_SITECUSTOMIZE: Attempting to import verl.workers.fsdp_workers")
        print("DEBUG_SITECUSTOMIZE: Attempting to import verl.workers.fsdp_workers")
        
        wrk_mod = import_module("verl.workers.fsdp_workers")
        Worker = getattr(wrk_mod, "ActorRolloutRefWorker")
        
        _log_to_proof_file(f"DEBUG_SITECUSTOMIZE: Found Worker: {Worker}")
        print(f"DEBUG_SITECUSTOMIZE: Found Worker: {Worker}")

        if not getattr(Worker, "_spectrum_patched", False):
            _log_to_proof_file("DEBUG_SITECUSTOMIZE: Worker not patched yet. Proceeding with patching init_model.")
            print("DEBUG_SITECUSTOMIZE: Worker not patched yet. Proceeding with patching init_model.")
            
            orig_init_model = Worker.init_model

            def patched_init_model(self, *args, **kwargs):
                _log_to_proof_file(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Inside patched_init_model for worker {self}. PID: {os.getpid()}")
                print(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Inside patched_init_model for worker {self}. PID: {os.getpid()}")
                
                orig_init_model(self, *args, **kwargs)  # Call original init_model
                
                if self._is_actor and hasattr(self, "actor_module_fsdp"):
                    _log_to_proof_file(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Worker {self} is actor and has actor_module_fsdp. Applying Spectrum freezing. PID: {os.getpid()}")
                    print(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Worker {self} is actor and has actor_module_fsdp. Applying Spectrum freezing. PID: {os.getpid()}")
                    
                    from verl_spectrum_patch.freezer import apply_spectrum_freezing
                    
                    self.actor_module_fsdp = apply_spectrum_freezing(self.actor_module_fsdp, YAML)
                    
                    trainable_params = sum(p.numel() for p in self.actor_module_fsdp.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.actor_module_fsdp.parameters())
                    
                    _log_to_proof_file(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Spectrum freezing applied. Trainable params: {trainable_params}, Total params: {total_params}. PID: {os.getpid()}")
                    print(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Spectrum freezing applied. Trainable params: {trainable_params}, Total params: {total_params}. PID: {os.getpid()}")
                else:
                    _log_to_proof_file(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Worker {self} is NOT actor or NO actor_module_fsdp. PID: {os.getpid()}")
                    print(f"DEBUG_SITECUSTOMIZE_PATCH_APPLIED: Worker {self} is NOT actor or NO actor_module_fsdp. PID: {os.getpid()}")


            Worker.init_model = patched_init_model
            Worker._spectrum_patched = True # Mark as patched
            
            _log_to_proof_file("DEBUG_SITECUSTOMIZE: Successfully patched Worker.init_model and set _spectrum_patched flag.")
            print("DEBUG_SITECUSTOMIZE: Successfully patched Worker.init_model and set _spectrum_patched flag.")
        else:
            _log_to_proof_file("DEBUG_SITECUSTOMIZE: Worker already patched (_spectrum_patched is True). Skipping.")
            print("DEBUG_SITECUSTOMIZE: Worker already patched (_spectrum_patched is True). Skipping.")

    except Exception as e:
        error_message = f"ERROR_SITECUSTOMIZE: Failed during sitecustomize patch: {e}\nTraceback: {import traceback; traceback.format_exc()}"
        _log_to_proof_file(error_message)
        print(error_message)
        # Optionally re-raise or handle as needed
        # raise
else:
    if not YAML:
        _log_to_proof_file("DEBUG_SITECUSTOMIZE: SPECTRUM_YAML_PATH not set. Patch not applied.")
        print("DEBUG_SITECUSTOMIZE: SPECTRUM_YAML_PATH not set. Patch not applied.")
    elif not os.path.exists(YAML):
        _log_to_proof_file(f"DEBUG_SITECUSTOMIZE: SPECTRUM_YAML_PATH is set to '{YAML}', but file does not exist. Patch not applied.")
        print(f"DEBUG_SITECUSTOMIZE: SPECTRUM_YAML_PATH is set to '{YAML}', but file does not exist. Patch not applied.")

_log_to_proof_file("DEBUG_SITECUSTOMIZE: sitecustomize.py execution finished.")
print("DEBUG_SITECUSTOMIZE: sitecustomize.py execution finished.") 