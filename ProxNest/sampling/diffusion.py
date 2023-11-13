import numpy as np
import torch
import torch.nn as nn
import deepinv



class ReflectedDiffPIR(nn.Module):
    r"""
    Reflected diffusion PnP Image Restoration (DiffPIR).

    This class implements the Diffusion PnP image restoration algorithm (DiffPIR) described
    in https://arxiv.org/abs/2305.08995.

    The DiffPIR algorithm is inspired on a half-quadratic splitting (HQS) plug-and-play algorithm, where the denoiser
    is a conditional diffusion denoiser, combined with a diffusion process. The algorithm writes as follows,
    for :math:`t` decreasing from :math:`T` to :math:`1`:

     .. math::
             \begin{equation*}
             \begin{aligned}
             x_{0}^{t} &= D_{\theta}(x_t, \frac{\sqrt{1-\overline{\alpha}_t}}{\sqrt{\overline{\alpha}_t}}) \\
             \widehat{x}_{0}^{t} &= \operatorname{prox}_{2 f(y, \cdot) /{\rho_t}}(x_{0}^{t}) \\
             \widehat{\varepsilon} &= \left(x_t - \sqrt{\overline{\alpha}_t} \,\,
             \widehat{x}_{0}^t\right)/\sqrt{1-\overline{\alpha}_t} \\
             \varepsilon_t &= \mathcal{N}(0, \mathbf{I}) \\
             x_{t-1} &= \sqrt{\overline{\alpha}_t} \,\, \widehat{x}_{0}^t + \sqrt{1-\overline{\alpha}_t}
             \left(\sqrt{1-\zeta} \,\, \widehat{\varepsilon} + \sqrt{\zeta} \,\, \varepsilon_t\right),
             \end{aligned}
             \end{equation*}

    where :math:`D_\theta(\cdot,\sigma)` is a Gaussian denoiser network with noise level :math:`\sigma`
    and :math:`f(y, \cdot)` is the data fidelity
    term.

    .. note::

            The algorithm might require careful tunning of the hyperparameters :math:`\lambda` and :math:`\zeta` to
            obtain optimal results.

    :param torch.nn.Module model: a conditional noise estimation model
    :param float sigma: the noise level of the data
    :param deepinv.optim.DataFidelity data_fidelity: the data fidelity operator
    :param int max_iter: the number of iterations to run the algorithm (default: 100)
    :param float zeta: hyperparameter :math:`\zeta` for the sampling step (must be between 0 and 1). Default: 1.0.
    :param float lambda_: hyperparameter :math:`\lambda` for the data fidelity step
        (:math:`\rho_t = \lambda \frac{\sigma_n^2}{\bar{\sigma}_t^2}` in the paper where the optimal value range
         between 3.0 and 25.0 depending on the problem). Default: 7.0.
    :param dict diff_params: dictionary with the `reflection_pos` and `reflection_strategy` parameters.
    :param bool verbose: if True, print progress
    :param str device: the device to use for the computations
    """

    def __init__(
        self,
        model,
        data_fidelity,
        boundary_indicator,
        sigma=0.05,
        max_iter=100,
        zeta=1.0,
        lambda_=7.0,
        diff_params=None,
        verbose=False,
        device="cpu",
    ):
        super(ReflectedDiffPIR, self).__init__()
        self.model = model
        self.lambda_ = lambda_
        self.data_fidelity = data_fidelity
        self.boundary_indicator = boundary_indicator
        self.max_iter = max_iter
        self.zeta = zeta
        self.diff_params = diff_params
        if diff_params is None or (
            "reflection_pos" not in diff_params or "reflection_strategy" not in diff_params
        ):
            self.reflection_pos = 'beggining'
            self.reflection_strategy = 1
        else:
            self.reflection_pos = self.diff_params["reflection_pos"]
            self.reflection_strategy = self.diff_params["reflection_strategy"]

        self.verbose = verbose
        self.device = device
        self.beta_start, self.beta_end = 0.1 / 1000, 20 / 1000
        self.num_train_timesteps = 1000

        (
            self.sqrt_1m_alphas_cumprod,
            self.reduced_alpha_cumprod,
            self.sqrt_alphas_cumprod,
            self.sqrt_recip_alphas_cumprod,
            self.sqrt_recipm1_alphas_cumprod,
            self.betas,
        ) = self.get_alpha_beta()

        self.rhos, self.sigmas, self.seq = self.get_noise_schedule(sigma=sigma)

    def get_alpha_beta(self):
        """
        Get the alpha and beta sequences for the algorithm. This is necessary for mapping noise levels to timesteps.
        """
        betas = np.linspace(
            self.beta_start, self.beta_end, self.num_train_timesteps, dtype=np.float32
        )
        betas = torch.from_numpy(betas).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t

        # Useful sequences deriving from alphas_cumprod
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        reduced_alpha_cumprod = torch.div(
            sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod
        )  # equivalent noise sigma on image
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        return (
            sqrt_1m_alphas_cumprod,
            reduced_alpha_cumprod,
            sqrt_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            betas,
        )

    def get_noise_schedule(self, sigma):
        """
        Get the noise schedule for the algorithm.
        """
        lambda_ = self.lambda_
        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(self.num_train_timesteps):
            sigmas.append(self.reduced_alpha_cumprod[self.num_train_timesteps - 1 - i])
            sigma_ks.append(
                (self.sqrt_1m_alphas_cumprod[i] / self.sqrt_alphas_cumprod[i])
            )
            rhos.append(lambda_ * (sigma**2) / (sigma_ks[i] ** 2))
        rhos, sigmas = torch.tensor(rhos).to(self.device), torch.tensor(sigmas).to(
            self.device
        )

        seq = np.sqrt(np.linspace(0, self.num_train_timesteps**2, self.max_iter))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1

        return rhos, sigmas, seq

    def find_nearest(self, array, value):
        """
        Find the argmin of the nearest value in an array.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def get_alpha_prod(
        self, beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=1000
    ):
        """
        Get the alpha sequences; this is necessary for mapping noise levels to timesteps when performing pure denoising.
        """
        betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        betas = torch.from_numpy(
            betas
        )  # .to(self.device) Removing this for now, can be done outside
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t

        # Useful sequences deriving from alphas_cumprod
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        return (
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
        )

    
    def reflect(self, x, y, physics):
        """ Reflect on the boundary

        """
        if self.reflection_strategy == 1:
            # Compute projection based on the data fidelity prox
            x_proj = x / 2 + 0.5
            x_proj = self.data_fidelity.prox(
                x_proj, y, physics
            )
            x_proj = x_proj * 2 - 1

            # Reflection step (based on projection operator)
            x = x + 2 * (x_proj - x.clone())
        else:
            raise NotImplementedError('Reflection strategy requested is not implemented.')

        return x


    def forward(
        self,
        y,
        physics: deepinv.physics.LinearPhysics,
        seed=None,
        x_init=None,
    ):
        r"""
        Runs the diffusion to obtain a random sample of the posterior distribution.

        :param torch.Tensor y: the measurements.
        :param deepinv.physics.LinearPhysics physics: the physics operator.
        :param float sigma: the noise level of the data.
        :param int seed: the seed for the random number generator.
        :param torch.Tensor x_init: the initial guess for the reconstruction.
        """

        if seed:
            torch.manual_seed(seed)

        if hasattr(physics.noise_model, "sigma"):
            sigma = physics.noise_model.sigma  # Then we overwrite the default values
            self.rhos, self.sigmas, self.seq = self.get_noise_schedule(sigma=sigma)

        # Initialization
        if x_init is None:  # Necessary when x and y don't live in the same space
            x = 2 * physics.A_adjoint(y) - 1
        else:
            x = 2 * x_init - 1

        sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod = self.get_alpha_prod()

        with torch.no_grad():
            for i in range(len(self.seq)):
                # Current noise level
                curr_sigma = self.sigmas[self.seq[i]].cpu().numpy()

                # time step associated with the noise level sigmas[i]
                t_i = self.find_nearest(self.reduced_alpha_cumprod, curr_sigma)

                # Denoising step
                x_aux = x / 2 + 0.5
                denoised = 2 * self.model(x_aux, curr_sigma / 2) - 1
                noise_est = (
                    sqrt_recip_alphas_cumprod[t_i] * x - denoised
                ) / sqrt_recipm1_alphas_cumprod[t_i]

                x0 = (
                    self.sqrt_recip_alphas_cumprod[t_i] * x
                    - self.sqrt_recipm1_alphas_cumprod[t_i] * noise_est
                )
                x0 = x0.clamp(-1, 1)

                if not self.seq[i] == self.seq[-1]:
                    # SKIP -> Data fidelity step

                    if self.reflection_pos == "beggining":
                        # Check if the x0 is outside the boundary
                        if not self.boundary_indicator(x0):
                            x0 = self.reflect(x0.clone(), y, physics)

                    # Sampling step
                    t_im1 = self.find_nearest(
                        self.reduced_alpha_cumprod,
                        self.sigmas[self.seq[i + 1]].cpu().numpy(),
                    )  # time step associated with the next noise level
                    eps = (
                        x - self.sqrt_alphas_cumprod[t_i] * x0
                    ) / self.sqrt_1m_alphas_cumprod[
                        t_i
                    ]  # effective noise
                    if self.zeta is not None:
                        x = (
                            self.sqrt_alphas_cumprod[t_im1] * x0
                            + self.sqrt_1m_alphas_cumprod[t_im1]
                            * np.sqrt(1 - self.zeta)
                            * eps
                            + self.sqrt_1m_alphas_cumprod[t_im1]
                            * np.sqrt(self.zeta)
                            * torch.randn_like(x)
                        )  # sampling (eq15)
                    else:
                        sigma_t = (
                            self.sqrt_1m_alphas_cumprod[t_im1] / self.sqrt_1m_alphas_cumprod[t_i]
                            ) * (self.sqrt_1m_alphas_cumprod[t_i] / self.sqrt_alphas_cumprod[t_i])
                        x = (
                            self.sqrt_alphas_cumprod[t_im1] * x0
                            + np.sqrt(self.sqrt_1m_alphas_cumprod[t_im1]**2 - sigma_t**2) * eps
                            + sigma_t * torch.randn_like(x)
                        ) # sampling (eq14)

                        if self.reflection_pos == "end":
                            # Check if the x0 is outside the boundary
                            if not self.boundary_indicator(x):
                                x = self.reflect(x.clone(), y, physics)

        out = x / 2 + 0.5  # back to [0, 1] range

        return out