import importlib
from pathlib import Path
cur_dir = Path(__file__).parent

lvu_init_model_map = {}
lvu_run_model_map = {}

for file in cur_dir.glob("*.py"):
    if file.name == "__init__.py":
        continue
    module_name = file.stem
    module = importlib.import_module(f".{module_name}", package=__package__)
    assert hasattr(module, "init_lvu_model"), f"Module {module_name} does not have init_lvu_model function."
    assert hasattr(module, "run_lvu_model"), f"Module {module_name} does not have run_lvu_model function."
    lvu_init_model_map[module_name] = module.init_lvu_model
    lvu_run_model_map[module_name] = module.run_lvu_model

__all__ = []
for module_name in lvu_init_model_map.keys():
    __all__.append(module_name)
    
__all__.append("lvu_init_model_map")
__all__.append("lvu_run_model_map")
