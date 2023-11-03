import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
sns.set()


def main(config_path, device):
    # I need to import these packages after selecting the GPU to be used
    import deepinv as dinv
    import ProxNest as pxn

    cfg = pxn.utils.utils.load_config(config_path)

    # Load Image
    x_true = np.load(cfg.input_data_dir + 'butterfly_{}.npy'.format(cfg.img_size))
    # Normalise magnitude
    x_true -= np.nanmin(x_true)
    x_true /= np.nanmax(x_true)
    x_true[x_true<0] = 0
    # To pytorch
    x_true = torch.Tensor(x_true).to(device)
    x = x_true.clone().to(device)


    # Define noise parameters
    if cfg.ISNR is not None:
        sigma = np.sqrt(np.mean(np.abs(x_true.cpu().numpy())**2)) * 10**(-cfg.ISNR/20)
    else:
        sigma = cfg.sigma
        if cfg.dtype == 'uint':
            sigma /= 255

    # Define the forward operator
    physics = dinv.physics.BlurFFT(
        img_size=(3, x.shape[-2], x.shape[-1]),
        filter=torch.ones(
            (1, 1, int(cfg.blur_kernel_size), int(cfg.blur_kernel_size)),
            device=device
        ) / cfg.blur_kernel_size**2,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )

    # Compute observations
    y = physics(x)

    # Options dictionary associated with the overall sampling algorithm
    options = {
                   'samplesL' : cfg.samplesL,               # Number of live samples
                   'samplesD' : cfg.samplesD,               # Number of discarded samples 
                      'sigma' : sigma,                      # Noise standard deviation of degraded image
                        'tol' : float(cfg.tol),             # Convergence tolerance of algorithm (Ball projection alg)
                   'max_iter' : cfg.max_iter,               # Maximum number of iterations (Ball projection alg)
                    'verbose' : cfg.verbose,                # Verbosity
                       'ISNR' : cfg.ISNR,                   # Input SNR
                    'img_size': cfg.img_size,               # Input image size
                  'wandb_vis' : cfg.wandb_vis,              # Use wandb visualisation logger
             'wandb_vis_imgs' : cfg.wandb_vis_imgs,         # Use wandb to visualise imgs
        'wandb_vis_imgs_freq' : cfg.wandb_vis_imgs_freq,    # Frequency to upload img to wandb
                 'experiment' : cfg.experiment,             # Experiment name
                   'run_name' : cfg.run_name,               # Run name
    }

    diff_params = {
             'model_type' : cfg.model_type,         # Type of pretrained model. Options are 'imagenet' or 'ffhq'
            'in_channels' : cfg.in_channels,        # Channels in the input Tensor.
           'out_channels' : cfg.out_channels,       # Channels in the output Tensor.
            'sigma_noise' : sigma,                  # Noise standard deviation of degraded image
        'diffusion_steps' : cfg.diffusion_steps,    # Maximum number of iterations of the DiffPIR algorithm
                'lambda_' : cfg.lambda_,            # Regularisation parameter
                   'zeta' : cfg.zeta,               # DiffPIR parameter controling the diffusion
    }

    # Gaussian log likelihood
    LogLikeliL = lambda x_current, y, physics, sigma : - torch.nn.functional.mse_loss(
        y,
        physics.A(x_current), # Apply the forward model (without the noise addition)
        reduction='sum'
    ) / (2*sigma**2)

    # Load the denoiser for the diffusion model 
    if diff_params['model_type'] == 'imagenet':
        model_path = "/disk/xray99/tl3/pretrained_diffusions/diffpir_pretrained_models/256x256_diffusion_uncond.pt"
        large_model = True
    elif diff_params['model_type'] == 'ffhq':
        model_path = "/disk/xray99/tl3/pretrained_diffusions/diffpir_pretrained_models/diffusion_ffhq_10m.pt"
        large_model = False

    denoising_model = dinv.models.DiffUNet(
        in_channels=diff_params['in_channels'],
        out_channels=diff_params['out_channels'],
        pretrained=model_path,
        large_model=large_model
    ).to(device)


    # Initialise x
    x_init = physics.A_adjoint(y)
    # Initialise diffnest model
    diffnest = pxn.sampling.diff_nested.DiffusionNestedSampling(
        x_init=x_init,
        y=y,
        denoising_model=denoising_model,
        physics=physics,
        LogLikeliL=LogLikeliL,
        options=options,
        diff_params=diff_params,
        device=device
    )

    # Run
    BayEvi, Xtrace = diffnest.run()


    # Plot likelihood evolution and save it
    plt.figure(figsize=(10, 6))
    plt.plot(
        -Xtrace["DiscardL"],
        alpha=0.75,
        label='Discarded likelihood'
    )
    plt.plot(
        - np.ones_like(Xtrace["DiscardL"]) * diffnest.x_sample_init_logLikeL,
        '--',
        alpha=0.75,
        label='Diffusion init sample likelihood'
    )
    plt.legend()
    plt.xlabel('Discarded sample number')
    plt.ylabel('- log Likelihood')
    plt.savefig(cfg.save_dir + cfg.experiment + cfg.run_name + 'likelihood_evolution.pdf')
    plt.show()

    # Save results
    save_dict = {
        'BayEvi': BayEvi,
        'Xtrace': Xtrace,
    }
    np.save(
        cfg.save_dir + cfg.experiment + cfg.run_name + 'nest_diff_variables.npy',
        save_dict,
        allow_pickle=True
    )



if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch nested sampling')
    parser.add_argument(
        '-c', '--config',
        default="./../configs/diffnest_DiffPIR_blur_64.yml",
        type=str,
        help='path to config file'
    )
    parser.add_argument(
        '-g', '--gpuid',
        default=0,
        type=int,
        help='GPU ID to use'
    )
    args = parser.parse_args()

    M1 = False
    if M1:
        device = torch.device(
            'mps' if torch.backends.mps.is_available() else 'cpu'
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuid)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(torch.cuda.is_available())
            print(torch.cuda.device_count())
            print(torch.cuda.current_device())
            print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # Train model
    main(args.config, device)
