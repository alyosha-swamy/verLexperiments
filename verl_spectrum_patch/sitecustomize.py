import os, logging
from importlib import import_module

log = logging.getLogger(__name__)
YAML = os.getenv("SPECTRUM_YAML_PATH")

if YAML and os.path.exists(YAML):
    try:
        wrk_mod = import_module("verl.workers.fsdp_workers")
        Worker   = getattr(wrk_mod, "ActorRolloutRefWorker")
        if not getattr(Worker, "_spectrum_patched", False):
            orig = Worker.init_model
            def patched(self, *a, **kw):
                orig(self, *a, **kw)           # original build
                if self._is_actor and hasattr(self, "actor_module_fsdp"):
                    log.info("Applying Spectrum parameter freezing via sitecustomize.py...")
                    from verl_spectrum_patch.freezer import apply_spectrum_freezing
                    self.actor_module_fsdp = apply_spectrum_freezing(self.actor_module_fsdp, YAML)
                    tr = sum(p.numel() for p in self.actor_module_fsdp.parameters() if p.requires_grad)
                    tot = sum(p.numel() for p in self.actor_module_fsdp.parameters())
                    log.info(f"Spectrum: {tr:,}/{tot:,} params trainable ({ (tr/tot)*100 if tot > 0 else 0 :.2f}%)")
            Worker.init_model = patched
            Worker._spectrum_patched = True
            log.info("Spectrum freezer activated via sitecustomize.py")
    except Exception as e:
        log.error(f"Error in Spectrum sitecustomize patch: {e}")
        import traceback
        traceback.print_exc()
else:
    log.warning("SPECTRUM_YAML_PATH unset or file not found --> skipping freezer patch in sitecustomize.py") 