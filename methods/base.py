from __future__ import annotations
from typing import Any, Dict, Optional, Protocol
from dataclasses import dataclass, field
import torch

Tensor = torch.Tensor


@dataclass
class Obs:
    """
    Container for measurements and operator-specific metadata.

    Examples:
        - inpainting: mask
        - deblurring: kernel
        - super-resolution: blur kernel, scale factor
        - MRI: mask, sensitivity maps
        - CT: geometry parameters
    """
    y: Tensor
    aux: Dict[str, Any] = field(default_factory=dict)

class InverseProblem(Protocol):
    """
    Generic interface for inverse problems of the form

        y = A(x) + noise

    used inside a RED-DEQ block.
    """
    name: str

    def make_observation(self, x_gt: Tensor) -> Obs:
        """
        Generate observation y (and any auxiliary metadata) from ground truth x_gt.
        """
        ...

    def forward(self, x: Tensor, obs: Obs) -> Tensor:
        """
        Apply the forward operator A(x).
        """
        ...

    def adjoint(self, y: Tensor, obs: Obs) -> Tensor:
        """
        Apply the adjoint operator A^T(y).
        """
        ...

    def normal(self, x: Tensor, obs: Obs) -> Tensor:
        """
        Apply the normal operator A^T A x.

        Default implementations in concrete classes can simply return:
            self.adjoint(self.forward(x, obs), obs)
        but some problems may override this for efficiency.
        """
        ...

    def solve_normal_plus_lambda(
        self,
        rhs: Tensor,
        lam: Tensor,
        obs: Obs,
        x0: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Solve

            (A^T A + lam * I) x = rhs

        where lam may be batch-shaped and broadcastable to rhs.

        This is the key problem-specific linear solve needed by RED-DEQ.

        Examples:
            - inpainting: closed form
            - deblurring: FFT or CG
            - MRI: CG
            - CT: usually iterative solver
        """
        ...

    def project(self, x: Tensor, obs: Obs) -> Tensor:
        """
        Optional hard projection / exact data-consistency step for inference-time use.

        If not needed, concrete implementations may simply return x.
        """
        ...