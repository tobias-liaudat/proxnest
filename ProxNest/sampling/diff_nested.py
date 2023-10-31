import numpy as np
import torch
from tqdm import tqdm
import ProxNest.logs as log
from . import resampling
import deepinv as dinv


class DummyDataFidelity(torch.nn.Module):
    def __init__(self):
        self.a = 1

    def prox(self, x, y=None, physics=None, gamma=None):
        return x


class DiffusionNestedSampling(torch.nn.Module):
    r"""
    Nested sampling with diffusion models in pytorch.
    """
    def __init__(
        self,
        x_init,
        y,
        denoising_model,
        physics,
        LogLikeliL,
        options,
        diff_params,
        device='cpu'
    ):
        super(DiffusionNestedSampling, self).__init__()
        self.y = y
        self.physics = physics
        self.denoising_model = denoising_model
        self.LogLikeliL = LogLikeliL
        self.options = options
        self.diff_params = diff_params
        self.device = device

        # Set initial state as current state
        self.x_init = x_init
        self.Xcur = x_init
        tau_0 = -self.LogLikeliL(
            self.Xcur, self.y, self.physics, self.diff_params['sigma_noise']
        ).cpu().numpy() * 1e-1

        # Initialise arrays to store samples
        # Indexing: sample, likelihood, weights
        self.NumLiveSetSamples = int(options["samplesL"])
        self.NumDiscardSamples = int(options["samplesD"])

        # Placeholder for Bayesian evidence
        self.BayEvi = np.zeros(2)

        self.Xtrace = {}

        # torch images are (1 x C x H x W)
        self.Xtrace["LiveSet"] = torch.zeros(
            (
                self.NumLiveSetSamples,
                1,
                self.Xcur.shape[1],
                self.Xcur.shape[2],
                self.Xcur.shape[3]
            ), # Batch x 1 x Channels x Heigth x Width
            dtype=self.Xcur.dtype,
            device=self.device,
            requires_grad=False
        )
        self.Xtrace["LiveSetL"] = np.zeros(self.NumLiveSetSamples)

        self.Xtrace["Discard"] = torch.zeros(
            (
                self.NumDiscardSamples,
                1,
                self.Xcur.shape[1],
                self.Xcur.shape[2],
                self.Xcur.shape[3]
            ), # Batch x 1 x Channels x Heigth x Width
            dtype=self.Xcur.dtype,
            device=self.device,
            requires_grad=False
        )
        self.Xtrace["DiscardL"] = np.zeros(self.NumDiscardSamples)
        self.Xtrace["DiscardW"] = np.zeros(self.NumDiscardSamples)
        self.Xtrace["DiscardPostProb"] = np.zeros(self.NumDiscardSamples)

        # Build prior sampler
        self.prior_sampler = dinv.sampling.DiffPIR(
            self.denoising_model,
            sigma=self.diff_params['sigma_noise'],
            max_iter=self.diff_params['diffusion_steps'],
            lambda_=self.diff_params['lambda_'],
            zeta=self.diff_params['zeta'], # 0.5,
            data_fidelity=DummyDataFidelity(),
            verbose=False,
            device=device
        )
        # How to sample from the prior
        # xhat =  self.prior_sampler.forward(y, physics, x_init)

        # Build constrained prior sampler
        self.constrained_prior_sampler = dinv.sampling.DiffPIR(
            self.denoising_model,
            sigma=self.diff_params['sigma_noise'],
            max_iter=self.diff_params['diffusion_steps'],
            lambda_=self.diff_params['lambda_'],
            zeta=self.diff_params['zeta'],
            data_fidelity=dinv.optim.data_fidelity.IndicatorL2(
                radius=(np.sqrt(
                    tau_0 * 2 * self.diff_params['sigma_noise']**2
                )).astype(np.float32)
            ),
            verbose=self.options['verbose'],
            device=device
        )


    def update_likelihood_constraint(self, tau):
        """ Update constrained prior sampler with new likelihood constraint.
        """
        self.constrained_prior_sampler = dinv.sampling.DiffPIR(
            self.denoising_model,
            sigma=self.diff_params['sigma_noise'],
            max_iter=self.diff_params['diffusion_steps'],
            lambda_=self.diff_params['lambda_'],
            zeta=self.diff_params['zeta'],
            data_fidelity=dinv.optim.data_fidelity.IndicatorL2(
                radius=(np.sqrt(
                    tau * 2 * self.diff_params['sigma_noise']**2
                )).astype(np.float32)
            ),
            verbose=self.options['verbose'],
            device=self.device
        )


    def init_live_samples(self):
        # Compute initialisation based on the observations

        # Obtain samples from priors
        for j in tqdm(range(self.NumLiveSetSamples), desc="DiffNest || Populate"):
            with torch.no_grad():
                # Sample from the prior to generate live samples
                self.Xcur = self.prior_sampler.forward(
                    self.y, self.physics, x_init=self.x_init
                )
                # Record the current sample in the live set and its likelihood
                self.Xtrace["LiveSet"][j] = self.Xcur.clone()
                self.Xtrace["LiveSetL"][j] = self.LogLikeliL(
                    self.Xcur, self.y, self.physics, self.diff_params['diffusion_steps']
                ).detach().cpu().numpy()


    def evolve_samples(self):

        # Update samples using the proximal nested sampling technique
        for k in tqdm(range(self.NumDiscardSamples), desc="DiffNest || Sample"):

            # Reorder samples TODO: Make this more efficient!
            self.Xtrace["LiveSet"], self.Xtrace["LiveSetL"] = resampling.reorder_samples(
                self.Xtrace["LiveSet"], self.Xtrace["LiveSetL"]
            )

            # Compute the smallest threshold wrt live samples' likelihood
            tau = -self.Xtrace["LiveSetL"][-1]  # - 1e-2

            # Randomly select a sample in the live set as a starting point
            indNewSample = (
                np.floor(np.random.rand() * (self.NumLiveSetSamples - 1)).astype(int) - 1
            )
            self.Xcur = self.Xtrace["LiveSet"][indNewSample]

            # Update likelihood constraint
            self.update_likelihood_constraint(tau)

            # Sample from the constrained prior
            self.Xcur = self.constrained_prior_sampler.forward(
                self.y, self.physics, x_init=self.Xcur
            )

            # check if the new sample is inside l2-ball (metropolis-hasting); if
            # not, force the new sample into L2-ball
            if torch.nn.functional.mse_loss(
                self.y, self.physics.A(self.Xcur), reduction='sum'
            ) > (
                tau * 2 * self.diff_params['sigma_noise']**2
            ):
                print('Explicitly enforcing L2 ball in MH step.')
                indicatorL2 = dinv.optim.data_fidelity.IndicatorL2(
                    radius=np.sqrt(
                        tau * 2 * self.diff_params['sigma_noise']**2
                    ).astype(np.float32) 
                )
                self.Xcur = indicatorL2.prox(
                    x=self.Xcur,
                    y=self.y,
                    physics=self.physics,
                    crit_conv=self.options['tol'],
                    max_iter=self.options['max_iter'],
                )


            # Record the sample discarded and its likelihood
            self.Xtrace["Discard"][k] = self.Xtrace["LiveSet"][-1].clone()
            self.Xtrace["DiscardL"][k] = self.Xtrace["LiveSetL"][-1].copy()

            # Add the new sample to the live set and its likelihood
            self.Xtrace["LiveSet"][-1] = self.Xcur.clone()
            self.Xtrace["LiveSetL"][-1] = self.LogLikeliL(
                    self.Xcur, self.y, self.physics, self.diff_params['diffusion_steps']
                ).detach().cpu().numpy()
            



        # Reorder the live samples TODO: Make this more efficient!
        self.Xtrace["LiveSet"], self.Xtrace["LiveSetL"] = resampling.reorder_samples(
            self.Xtrace["LiveSet"], self.Xtrace["LiveSetL"]
        )

    def compute_evidence_stats(self):
        # Bayesian evidence calculation
        self.BayEvi = np.zeros(2)
        self.Xtrace["DiscardW"][0] = 1 / self.NumLiveSetSamples

        # Compute the sample weight
        for k in tqdm(range(self.NumDiscardSamples), desc="DiffNest || Compute Weights"):
            self.Xtrace["DiscardW"][k] = np.exp(-(k + 1) / self.NumLiveSetSamples)

        # Compute the volumn length for each sample using trapezium rule
        discardLen = np.zeros(self.NumDiscardSamples)
        discardLen[0] = (1 - np.exp(-2 / self.NumLiveSetSamples)) / 2

        for i in tqdm(
            range(1, self.NumDiscardSamples - 1), desc="DiffNest || Trapezium Integrate"
        ):
            discardLen[i] = (self.Xtrace["DiscardW"][i - 1] - self.Xtrace["DiscardW"][i + 1]) / 2

        discardLen[-1] = (
            np.exp(-(self.NumDiscardSamples - 1) / self.NumLiveSetSamples)
            - np.exp(-(self.NumDiscardSamples + 1) / self.NumLiveSetSamples)
        ) / 2
        # volume length of the last discarded sample

        liveSampleLen = np.exp(-(self.NumDiscardSamples) / self.NumLiveSetSamples)
        # volume length of the living sample

        # Apply the disgarded sample for Bayesian evidence value computation
        vecDiscardLLen = self.Xtrace["DiscardL"] + np.log(discardLen)

        # Apply the final live set samples for Bayesian evidence value computation
        vecLiveSetLLen = self.Xtrace["LiveSetL"] + np.log(liveSampleLen / self.NumLiveSetSamples)

        # #   ------- Way 1: using discarded and living samples --------
        # # Get the maximum value of the exponents for all the samples
        # maxAllSampleLLen = max(max(vecDiscardLLen),max(vecLiveSetLLen))

        # # Compute the Bayesian evidence value using discarded and living samples
        # BayEvi[0] = maxAllSampleLLen + np.log(np.sum(np.exp(vecDiscardLLen-maxAllSampleLLen)) + np.sum(np.exp(vecLiveSetLLen-maxAllSampleLLen)))

        # ------- Way 2: using discarded samples --------
        # Get the maximum value of the exponents for the discarded samples
        maxDiscardLLen = np.max(vecDiscardLLen)

        # Compute the Bayesian evidence value using discarded and living samples
        self.BayEvi[0] = maxDiscardLLen + np.log(np.sum(np.exp(vecDiscardLLen - maxDiscardLLen)))

        # Extimate the error of the computed Bayesian evidence
        entropyH = 0

        for k in tqdm(range(self.NumDiscardSamples), desc="ProxNest || Estimate Variance"):
            temp1 = np.exp(self.Xtrace["DiscardL"][k] + np.log(discardLen[k]) - self.BayEvi[0])
            entropyH = entropyH + temp1 * (self.Xtrace["DiscardL"][k] - self.BayEvi[0])

        # Evaluate the evidence variance
        self.BayEvi[1] = np.sqrt(np.abs(entropyH) / self.NumLiveSetSamples)

        # Compute the posterior probability for each discarded sample
        for k in tqdm(range(self.NumDiscardSamples), desc="ProxNest || Compute Posterior Mean"):
            self.Xtrace["DiscardPostProb"][k] = np.exp(
                self.Xtrace["DiscardL"][k] + np.log(discardLen[k]) - self.BayEvi[0]
            )

        # Compute the posterior mean of the discarded samples -- optimal solution
        self.Xtrace["DiscardPostMean"] = torch.zeros(
            (
                1,
                self.Xcur.shape[1],
                self.Xcur.shape[2],
                self.Xcur.shape[3]
            ), # 1 x Channels x Heigth x Width
            dtype=self.Xcur.dtype,
            device=self.device,
            requires_grad=False
        )
        for k in range(self.NumDiscardSamples):
            self.Xtrace["DiscardPostMean"] = self.Xtrace["DiscardPostMean"] + (
                self.Xtrace["DiscardPostProb"][k] * self.Xtrace["Discard"][k].clone()
            )

    
    def run(self):

        self.init_live_samples()
        self.evolve_samples()
        self.compute_evidence_stats()

        return self.BayEvi, self.Xtrace

