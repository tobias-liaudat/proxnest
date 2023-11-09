
import numpy as np
from scipy.stats import multivariate_normal


# Customise a Multivariate Gaussian Class
class MultivariateGaussian:
    def __init__(self, mu, cov):
        """
        mu: 1d np array (D,)
        cov: 2d np array (D, D)
        """
        self.mu = mu
        self.cov = cov
        if np.abs(np.linalg.det(cov) )> 0 :
          self.prec = np.linalg.inv(cov )  # precision matrix
        else:
          self.prec = np.linalg.pinv(cov) # precision matrix
        self.RV = multivariate_normal(mu, cov, allow_singular=True)
        self.dim = len(mu)

    def pdf(self, x):
        """
        Compute probability density (PDF) at x.
        """
        return self.RV.pdf(x)

    def score(self, x):
        """
        Compute the score (gradient of log likelihood) for the given x.
        """
        return - (x - self.mu) @ self.prec

    def sample(self, N=1):
        """
        Draw N samples from the Gaussian.
        """
        return self.RV.rvs(N)
    

# Let's define a function diffuse_mgm_vp that turn the mgm at $t=0$ 
# into a mgm at  $t=t'$
def diffuse_mgm_vp(mgm1, t, Lambda):
  alpha_t = np.exp(-Lambda*t)  # variance
  mu_dif=np.sqrt(alpha_t)*mgm1.mu
  noise_cov = np.eye(mgm1.dim) * (1-alpha_t)
  cov_dif =alpha_t*mgm1.cov + noise_cov
  return MultivariateGaussian(mu_dif, cov_dif)


def reverse_diffusion_SDE_sampling_mgm_vp(
    mgm,
    sampN=1000,
    nsteps=2000,
    Lambda=1,
    xT=None
):
    """
    Using exact score function to simulate the reverse SDE with 
    beta(t) to sample from distribution.

    mgm: Mutivariate Gaussian Model
    sampN: Number of samples to generate
    nsteps: how many discrete steps do we use to simulate the process
    """
    # initial distribution $N(0,sigma_t_2(Lambda, t=1))I)$ .
    if xT is None :
      sigma_t_2= lambda Lambda , t :1-np.exp(-Lambda*t)
      xT = np.sqrt(sigma_t_2(Lambda,t=1))*np.random.randn(sampN, 2)
      x_traj_rev = np.zeros((*xT.shape, nsteps,))
      x_traj_rev[:, :, 0] = xT
    dt = 1 / nsteps
    for i in range(1, nsteps):
        # note the time fly back $t$
        t = 1 - i * dt

        # Sample the Gaussian noise $z ~ N(0, I)$
        z_t = np.random.randn(*xT.shape)

        # Transport the gmm to that at time $t$ and
        mgm_t = diffuse_mgm_vp(mgm, t, Lambda=1)

        # Compute the score at state $x_t$, $\nabla \log p_t(x_t)$
        score_xt = mgm_t.score(x_traj_rev[:, :, i-1])

        # Implement the one time step update equation with beta(t) = 1 for all t
        x_traj_rev[:, :, i] = (
           (1 + 0.5 * dt) * x_traj_rev[:, :, i-1]
        ) + (
           dt * score_xt - np.sqrt(dt) * z_t
        )
    return x_traj_rev

"""Example

#set_seed(42)
x_traj_rev_vp = reverse_diffusion_SDE_sampling_mgm_vp(
    mgm1, 
    sampN=100000,
    nsteps=2000,
    xT=x_traj[:,:,-1]
)
x0_rev_vp = x_traj_rev_vp[:, :, -1]
mgm_samples= mgm1.sample(100000)

with plt.xkcd():
    figh, axs = plt.subplots(1, 1, figsize=[6.5, 6])
    handles = []
    kdeplot(x0_rev_vp, "Samples from Reverse Diffusion VP", ax=axs, handles=handles)
    kdeplot(mgm_samples, "Samples from original mgm", ax=axs, handles=handles)
    gmm_pdf_contour_plot(mgm1, cmap='gray', levels=20)  # the exact pdf contour of gmm
    plt.legend(handles=handles)
    figh.show()

"""