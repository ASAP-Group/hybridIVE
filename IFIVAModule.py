import torch
import torch.nn as nn
from rfc_utils import robust_complex_divide

class IFIVAModule(nn.Module):
    """
    A standalone module for performing Iterative Fast Independent Vector Analysis (iFIVA)
    using a robust complex division method to prevent division by zero.
    """
    def __init__(self, num_iterations: int = 5, eps: float = 1e-8):
        super().__init__()
        self.num_iterations = num_iterations
        self.eps = eps
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")

    def forward(self, x: torch.Tensor, w_initial: torch.Tensor, a_initial: torch.Tensor,
                Cx: torch.Tensor, Cwinv: torch.Tensor, N: int):
        """
        Performs the iFIVA iterations using robust division.

        Args:
            x (torch.Tensor): The input complex signal (nbatches, K, d, N).
            w_initial (torch.Tensor): Initial separating matrix w (nbatches, K, d, 1).
            a_initial (torch.Tensor): Initial mixing matrix a (nbatches, K, d, 1).
            Cx (torch.Tensor): Covariance matrix of x (nbatches, K, d, d).
            Cwinv (torch.Tensor): Inverse of the weighted covariance matrix (nbatches, K, d, d).
            N (int): Number of samples/frames in the signal.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The final w and a after iterations.
        """
        w = w_initial
        a = a_initial
        
        # Pre-cast N to a tensor to avoid recreating it in the loop
        N_tensor = torch.tensor(float(N), device=x.device, dtype=torch.complex64)
        one_tensor = torch.tensor(1.0, device=x.device, dtype=torch.complex64)

        for _ in range(self.num_iterations):
            # Calculate source signals and their variance
            sigmax2 = torch.real(torch.sum(w.conj() * torch.matmul(Cx, w), dim=2, keepdim=True))
            sigma = torch.sqrt(torch.clamp(sigmax2, min=self.eps))
            soi = torch.matmul(w.conj().transpose(2, 3), x)
            soin = robust_complex_divide(soi, sigma, self.eps)

            # Calculate score function (psi)
            sp2 = torch.real(soin * soin.conj())
            aux_psi = 1.0 / (1.0 + torch.sum(sp2, dim=1, keepdim=True) + self.eps)
            psi = soin.conj() * aux_psi
            psihpsi = aux_psi - psi * psi.conj()

            # Update step
            rho = torch.mean(psihpsi, dim=3, keepdim=True)
            xpsi_numerator = torch.matmul(x, psi.transpose(2, 3))
            xpsi = robust_complex_divide(robust_complex_divide(xpsi_numerator, sigma, self.eps), N_tensor, self.eps)
            nu = torch.matmul(w.conj().transpose(2, 3), xpsi)
            
            grad = a - robust_complex_divide(xpsi, nu, self.eps)
            denom_for_a_update = (one_tensor - robust_complex_divide(rho, nu, self.eps))
            a = a - robust_complex_divide(grad, denom_for_a_update, self.eps)

            # Projection back to the sphere
            w = torch.matmul(Cwinv, a)
            sigmaw2_denom = torch.matmul(a.conj().transpose(2, 3), w)
            sigmaw2 = torch.real(robust_complex_divide(one_tensor, sigmaw2_denom, self.eps))
            w = w * sigmaw2
            a = torch.matmul(Cx, w)
            a_denom = torch.matmul(a.conj().transpose(2, 3), w)
            a = robust_complex_divide(a, a_denom, self.eps)

        return w, a