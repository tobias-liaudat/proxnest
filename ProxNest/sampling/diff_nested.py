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



def DiffusionNestedSampling_np(X0, LikeliL, proxH, proxB, params, options):
    r"""Executes the proximal nested sampling algorithm

    Args:
        X0 (np.ndarray): initialisation of the sample chain.

        LikeliL (lambda): function to compute the likelihood value of a sample.

        proxH (lambda): proximity operator of the prior.

        proxB (lambda): proximity operator of the constraint :math:`\ell_2`-ball.

        params (dict): parameters for prior resampling subject to likelihood isocontour.

        options (dict): parameters about number of samples, thinning factor, burnning numbers.

    Returns:
        tuple: (Evidence, sample trace).

    Notes:
        MATLAB version: Xiaohao Cai (21/02/2019)

        Python version: Matthew Price (9/05/2022)
    """
    sigma = options["sigma"]
    Phi = params["Phi"]
    y = params["y"]

    log.info_log("Constructing lambda functions for resampling projections...")

    # Simulation setup
    # Use backward-forward splitting to approximate proxPi using proxH and gradF
    # driftIniN = lambda X, delta, gamma: np.real(
    #     (1 - delta / (2 * gamma)) * X 
    #     + delta / (2 * gamma) * proxH(X, gamma)
    # )
    # drift = lambda X, delta, lamb, tau, gamma, sigma: np.real(
    #     (1 - delta / (2 * lamb) - delta / (2 * gamma)) * X
    #     + delta / (2 * gamma) * proxH(X, gamma)
    #     + delta / (2 * lamb) * proxB(X, np.sqrt(tau * 2 * sigma**2))
    # )

    prior_sampling = lambda x: x**2
    constrained_prior_sampling = lambda x: x**2


    # Initialize variables
    delta = options[
        "delta"
    ]  # delta controls the proposal variance, the step-length and Moreau approximation
    if options['lamb'] is None:
        lamb = 5 * delta  # lamb \in [4*delta, 10*delta]
    else:
        lamb = options['lamb']
    # If gamma not provided, copy the lamb value
    if options["gamma"] is None:
        gamma = lamb
    else:
        gamma = options["gamma"]
    Xcur = X0  # set initial state as current state
    tau_0 = -LikeliL(Xcur) * 1e-1

    log.info_log("Allocating memory and populating initial live-samples...")

    # Initialise arrays to store samples
    # Indexing: sample, likelihood, weights
    NumLiveSetSamples = options["samplesL"]
    NumDiscardSamples = options["samplesD"]

    Xtrace = {}

    Xtrace["LiveSet"] = np.zeros((NumLiveSetSamples, Xcur.shape[0], Xcur.shape[1]))
    Xtrace["LiveSetL"] = np.zeros(NumLiveSetSamples)

    Xtrace["Discard"] = np.zeros((NumDiscardSamples, Xcur.shape[0], Xcur.shape[1]))
    Xtrace["DiscardL"] = np.zeros(NumDiscardSamples)
    Xtrace["DiscardW"] = np.zeros(NumDiscardSamples)
    Xtrace["DiscardPostProb"] = np.zeros(NumDiscardSamples)

    # Compute initialisation based on the observations
    #TODO: x_init
    x_init = 1
    j = 0

    # Obtain samples from priors
    for ii in tqdm(range(1, NumLiveSetSamples), desc="DiffNest || Populate"):
        # Sample from the prior to generate live samples
        Xcur = prior_sampling(x_init)

        # Record the current sample in the live set and its likelihood
        Xtrace["LiveSet"][j] = Xcur
        Xtrace["LiveSetL"][j] = LikeliL(Xcur)

        j += 1

    log.info_log("Executing primary nested resampling iterations...")

    # Reorder samples TODO: Make this more efficient!
    Xtrace["LiveSet"], Xtrace["LiveSetL"] = resampling.reorder_samples(
        Xtrace["LiveSet"], Xtrace["LiveSetL"]
    )

    # Update samples using the proximal nested sampling technique
    for k in tqdm(range(NumDiscardSamples), desc="DiffNest || Sample"):
        # Compute the smallest threshold wrt live samples' likelihood
        tau = -Xtrace["LiveSetL"][-1]  # - 1e-2

        # Randomly select a sample in the live set as a starting point
        indNewSample = (
            np.floor(np.random.rand() * (NumLiveSetSamples - 1)).astype(int) - 1
        )
        Xcur = Xtrace["LiveSet"][indNewSample]

        # Sample from the constrained prior
        Xcur = constrained_prior_sampling(Xcur, np.sqrt(tau * 2 * sigma**2))

        # Generate a new sample with likelihood larger than given threshould
        Xcur = drift(
            Xcur, delta, lamb, tau, gamma, sigma
        ) + np.sqrt(delta) * np.random.randn(
            Xcur.shape[0], Xcur.shape[1]
        )

        # check if the new sample is inside l2-ball (metropolis-hasting); if
        # not, force the new sample into L2-ball
        if np.sum(np.sum(np.abs(y - Phi.dir_op(Xcur)) ** 2)) > tau * 2 * sigma**2:
            Xcur = proxB(Xcur, np.sqrt(tau * 2 * sigma**2))

        # Record the sample discarded and its likelihood
        Xtrace["Discard"][k] = Xtrace["LiveSet"][-1]
        Xtrace["DiscardL"][k] = Xtrace["LiveSetL"][-1]

        # Add the new sample to the live set and its likelihood
        Xtrace["LiveSet"][-1] = Xcur
        Xtrace["LiveSetL"][-1] = LikeliL(Xcur)

        # Reorder the live samples TODO: Make this more efficient!
        Xtrace["LiveSet"], Xtrace["LiveSetL"] = resampling.reorder_samples(
            Xtrace["LiveSet"], Xtrace["LiveSetL"]
        )

    log.info_log(
        "Estimating Bayesian evidence (with variance), posterior probabilies, and posterior mean..."
    )

    # Bayesian evidence calculation
    BayEvi = np.zeros(2)
    Xtrace["DiscardW"][0] = 1 / NumLiveSetSamples

    # Compute the sample weight
    for k in tqdm(range(NumDiscardSamples), desc="ProxNest || Compute Weights"):
        Xtrace["DiscardW"][k] = np.exp(-(k + 1) / NumLiveSetSamples)

    # Compute the volumn length for each sample using trapezium rule
    discardLen = np.zeros(NumDiscardSamples)
    discardLen[0] = (1 - np.exp(-2 / NumLiveSetSamples)) / 2

    for i in tqdm(
        range(1, NumDiscardSamples - 1), desc="ProxNest || Trapezium Integrate"
    ):
        discardLen[i] = (Xtrace["DiscardW"][i - 1] - Xtrace["DiscardW"][i + 1]) / 2

    discardLen[-1] = (
        np.exp(-(NumDiscardSamples - 1) / NumLiveSetSamples)
        - np.exp(-(NumDiscardSamples + 1) / NumLiveSetSamples)
    ) / 2
    # volume length of the last discarded sample

    liveSampleLen = np.exp(-(NumDiscardSamples) / NumLiveSetSamples)
    # volume length of the living sample

    # Apply the disgarded sample for Bayesian evidence value computation
    vecDiscardLLen = Xtrace["DiscardL"] + np.log(discardLen)

    # Apply the final live set samples for Bayesian evidence value computation
    vecLiveSetLLen = Xtrace["LiveSetL"] + np.log(liveSampleLen / NumLiveSetSamples)

    # #   ------- Way 1: using discarded and living samples --------
    # # Get the maximum value of the exponents for all the samples
    # maxAllSampleLLen = max(max(vecDiscardLLen),max(vecLiveSetLLen))

    # # Compute the Bayesian evidence value using discarded and living samples
    # BayEvi[0] = maxAllSampleLLen + np.log(np.sum(np.exp(vecDiscardLLen-maxAllSampleLLen)) + np.sum(np.exp(vecLiveSetLLen-maxAllSampleLLen)))

    # ------- Way 2: using discarded samples --------
    # Get the maximum value of the exponents for the discarded samples
    maxDiscardLLen = np.max(vecDiscardLLen)

    # Compute the Bayesian evidence value using discarded and living samples
    BayEvi[0] = maxDiscardLLen + np.log(np.sum(np.exp(vecDiscardLLen - maxDiscardLLen)))

    # Extimate the error of the computed Bayesian evidence
    entropyH = 0

    for k in tqdm(range(NumDiscardSamples), desc="ProxNest || Estimate Variance"):
        temp1 = np.exp(Xtrace["DiscardL"][k] + np.log(discardLen[k]) - BayEvi[0])
        entropyH = entropyH + temp1 * (Xtrace["DiscardL"][k] - BayEvi[0])

    # Evaluate the evidence variance
    BayEvi[1] = np.sqrt(np.abs(entropyH) / NumLiveSetSamples)

    # Compute the posterior probability for each discarded sample
    for k in tqdm(range(NumDiscardSamples), desc="ProxNest || Compute Posterior Mean"):
        Xtrace["DiscardPostProb"][k] = np.exp(
            Xtrace["DiscardL"][k] + np.log(discardLen[k]) - BayEvi[0]
        )

    # Compute the posterior mean of the discarded samples -- optimal solution
    Xtrace["DiscardPostMean"] = np.zeros((Xcur.shape[0], Xcur.shape[1]))
    for k in range(NumDiscardSamples):
        Xtrace["DiscardPostMean"] += Xtrace["DiscardPostProb"][k] * Xtrace["Discard"][k]

    return BayEvi, Xtrace
