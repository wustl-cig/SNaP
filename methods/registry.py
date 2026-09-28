from methods.inpainting import InpaintingProblem
from methods.denoising import DenoisingProblem
from methods.deblurring import GaussianDeblurringProblem
from methods.super_resolution import SuperResolutionProblem
from methods.mri import MRIProblem   # multi-coil CS-MRI

_PROBLEM_REGISTRY = {
    "inpainting": InpaintingProblem,      # random pixel inpainting (config: methods.inpainting)
    "box_inpainting": InpaintingProblem,  # centered box inpainting  (config: methods.box_inpainting)
    "denoising": DenoisingProblem,
    "blur_gauss": GaussianDeblurringProblem,
    "sr": SuperResolutionProblem,
    "cs_mri": MRIProblem,                 # multi-coil MRI (config: methods.cs_mri)
    "mri": MRIProblem,
}

def get_problem(name: str):
    """
    Returns the Problem class (not an instance).

    Example:
        ProblemCls = get_problem("inpainting")
        problem = ProblemCls(**kwargs)
    """
    if name not in _PROBLEM_REGISTRY:
        raise KeyError(
            f"Unknown problem '{name}'. "
            f"Available problems: {list(_PROBLEM_REGISTRY.keys())}"
        )
    return _PROBLEM_REGISTRY[name]
