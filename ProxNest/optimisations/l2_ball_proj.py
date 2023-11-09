import numpy as np
import torch
import deepinv as dinv


class CustomIndicatorL2(dinv.optim.data_fidelity.DataFidelity):
    r"""
    Indicator of :math:`\ell_2` ball with radius :math:`r`.

    The indicator function of the $\ell_2$ ball with radius :math:`r`, denoted as \iota_{\mathcal{B}_2(y,r)(u)},
    is defined as

    .. math::

          \iota_{\mathcal{B}_2(y,r)}(u)= \left.
              \begin{cases}
                0, & \text{if } \|u-y\|_2\leq r \\
                +\infty & \text{else.}
              \end{cases}
              \right.


    :param float radius: radius of the ball. Default: None.

    """

    def __init__(
        self,
        radius=None,
        projection_type='deepinv',
        sopt_params=None,
        max_iter=100,
        stepsize=None,
        crit_conv=1e-5
    ):
        super().__init__()
        self.radius = radius
        self.projection_type = projection_type
        self.sopt_params = sopt_params
        # Deepinv projection params
        self.max_iter = max_iter
        self.stepsize = stepsize
        self.crit_conv = crit_conv

        # Select the chosen projection type
        if self.projection_type == 'deepinv':
            self.prox = self._deepinv_prox
        elif self.projection_type == 'sopt':
            self.prox = self.sopt_fast_proj_B2

    def d(self, u, y, radius=None):
        r"""
        Computes the batched indicator of :math:`\ell_2` ball with radius `radius`, i.e. :math:`\iota_{\mathcal{B}(y,r)}(u)`.

        :param torch.tensor u: Variable :math:`u` at which the indicator is computed. :math:`u` is assumed to be of shape (B, ...) where B is the batch size.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`u`.
        :param float radius: radius of the :math:`\ell_2` ball. If `radius` is None, the radius of the ball is set to `self.radius`. Default: None.
        :return: (torch.tensor) indicator of :math:`\ell_2` ball with radius `radius`. If the point is inside the ball, the output is 0, else it is 1e16.
        """
        diff = u - y
        dist = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=-1)
        radius = self.radius if radius is None else radius
        loss = (dist > radius) * 1e16
        return loss

    def prox_d(self, x, y, radius=None, gamma=None):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\iota_{\mathcal{B}_2(y,r)}}(x) = \operatorname{proj}_{\mathcal{B}_2(y, r)}(x)


        where :math:`\operatorname{proj}_{C}(x)` denotes the projection on the closed convex set :math:`C`.


        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`x`.
        :param float gamma: step-size. Note that this parameter is not used in this function.
        :param float radius: radius of the :math:`\ell_2` ball.
        :return: (torch.tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius
        diff = x - y
        dist = torch.norm(diff.view(diff.shape[0], -1), p=2, dim=-1)
        return y + diff * (
            torch.min(torch.tensor([radius]).to(x.device), dist) / (dist + 1e-12)
        ).view(-1, 1, 1, 1)

    def _deepinv_prox(
        self,
        x,
        y,
        physics,
        radius=None,
        gamma=None,
    ):
        r"""
        Proximal operator of the indicator of :math:`\ell_2` ball with radius `radius`, i.e.

        .. math::

            \operatorname{prox}_{\gamma \iota_{\mathcal{B}_2(y, r)}(A\cdot)}(x) = \underset{u}{\text{argmin}} \,\, \iota_{\mathcal{B}_2(y, r)}(Au)+\frac{1}{2}\|u-x\|_2^2

        Since no closed form is available for general measurement operators, we use a dual forward-backward algorithm,
        as suggested in `Proximal Splitting Methods in Signal Processing <https://arxiv.org/pdf/0912.3522.pdf>`_.

        :param torch.tensor x: Variable :math:`x` at which the proximity operator is computed.
        :param torch.tensor y: Data :math:`y` of the same dimension as :math:`A(x)`.
        :param torch.tensor radius: radius of the :math:`\ell_2` ball.
        :param float stepsize: step-size of the dual-forward-backward algorithm.
        :param float crit_conv: convergence criterion of the dual-forward-backward algorithm.
        :param int max_iter: maximum number of iterations of the dual-forward-backward algorithm.
        :param float gamma: factor in front of the indicator function. Notice that this does not affect the proximity
                            operator since the indicator is scale invariant. Default: None.
        :return: (torch.tensor) projection on the :math:`\ell_2` ball of radius `radius` and centered in `y`.
        """
        radius = self.radius if radius is None else radius

        if physics.A(x).shape == x.shape and (physics.A(x) == x).all():  # Identity case
            return self.prox_d(x, y, gamma=None, radius=radius)
        else:
            norm_AtA = physics.compute_norm(x, verbose=False)
            stepsize = 1.0 / norm_AtA if self.stepsize is None else self.stepsize
            u = physics.A(x)
            for it in range(self.max_iter):
                u_prev = u.clone()

                t = x - physics.A_adjoint(u)
                u_ = u + stepsize * physics.A(t)
                u = u_ - stepsize * self.prox_d(
                    u_ / stepsize, y, radius=radius, gamma=None
                )
                rel_crit = ((u - u_prev).norm()) / (u.norm() + 1e-12)
                if rel_crit < self.crit_conv:
                    break
            return t.to(torch.float32)

    def sopt_fast_proj_B2(
            self,
            x,
            y,
            physics,
            radius=None,
            gamma=None,
        ):
        r"""Fast projection algorithm onto the :math:`\ell_2`-ball.

        Compute the projection onto the :math:`\ell_2` ball, i.e. solve

        .. math::

            z^* = \min_{z} ||x - z||_2^2   s.t.  ||y - \Phi z||_2 < \tau

        where :math:`x` is the input vector and the solution :math:`z^*` is returned as sol.

        Args:
            x (np.ndarray): A sample position :math:`x` in the posterior space.
            tau (float): Radius of likelihood :math:`\ell_2`-ball.
            params (dict): Dictionary of parameters defining the optimisation.

        Returns:
            np.ndarray: Optimal solution :math:`z^*` of proximal projection.

        Notes:
            [1] M.J. Fadili and J-L. Starck, 
                "Monotone operator splitting for optimization problems in sparse recovery",
                IEEE ICIP, Cairo, Egypt, 2009.
            [2] Amir Beck and Marc Teboulle,
                "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems", 
                SIAM Journal on Imaging Sciences 2 (2009), no. 1, 183--202.
        """
        # Update radius
        radius = self.radius if radius is None else radius
        # Define operators from the physics
        forward_op = lambda z: physics.A(z)
        adj_forward_op = lambda z: physics.A_adjoint(z)

        # Lambda function for scaling, used for tight frames only
        sc = lambda z: z * torch.minimum(radius / torch.linalg.norm(z), torch.ones_like(radius / torch.linalg.norm(z)))

        # TIGHT FRAMES
        if (self.sopt_params["tight"]) and (self.sopt_params["pos"] or self.sopt_params["reality"]):

            temp = forward_op(x) - y
            sol = x + 1 / self.sopt_params["nu"] * adj_forward_op(sc(temp) - temp)
            crit_B2 = "TOL_EPS"
            iter = 0
            u = 0

        # NON-TIGHT FRAMES
        else:

            # Initializations
            sol = x
            # u = params['u']
            u = forward_op(sol)
            v = u
            iter = 1
            told = 1

            # Tolerance onto the L2 ball
            epsilon_low = radius / (1 + self.sopt_params["tol"])
            epsilon_up = radius / (1 - self.sopt_params["tol"])

            # Check if we are in the L2 ball
            dummy = forward_op(sol)
            norm_res = torch.linalg.norm(y - dummy)
            if norm_res <= epsilon_up:
                crit_B2 = "TOL_EPS"
                true = 0

            # Projection onto the L2-ball
            if self.sopt_params["verbose"] > 1:
                print("  Proj. B2:")

            while 1:

                # Residual
                res = forward_op(sol) - y
                norm_res = torch.linalg.norm(res)

                # Scaling for the projection
                res = u * self.sopt_params["nu"] + res
                norm_proj = torch.linalg.norm(res)

                # Log
                if self.sopt_params["verbose"] > 1:
                    print(
                        "   Iter {}, epsilon = {}, ||y - Phi(x)||_2 = {}".format(
                            iter, radius, norm_res
                        )
                    )

                # Stopping criterion
                if (norm_res >= epsilon_low) and (norm_res <= epsilon_up):
                    crit_B2 = "TOL_EPS"
                    break
                elif iter >= self.sopt_params["max_iter"]:
                    crit_B2 = "MAX_IT"
                    break

                # Projection onto the L2 ball
                t = (1 + np.sqrt(1 + 4 * told**2)) / 2
                ratio = torch.minimum(torch.ones_like(radius / norm_proj), radius / norm_proj)
                u = v
                v = 1 / self.sopt_params["nu"] * (res - res * ratio)
                u = v + (told - 1) / t * (v - u)

                # Current estimate
                sol = x - adj_forward_op(u)

                # Projection onto the non-negative orthant (positivity constraint)
                if self.sopt_params["pos"]:
                    sol = sol.real
                    sol[sol < 0] = 0

                # Projection onto the real orthant (reality constraint)
                if self.sopt_params["reality"]:
                    sol = sol.real

                # Increment iteration labels
                told = t
                iter = iter + 1

        # Log after the projection onto the L2-ball
        if self.sopt_params["verbose"] >= 1:
            temp = forward_op(sol)
            print(
                "  Proj. B2: epsilon = {}, ||y - Phi(x)||_2 = {}, {}, iter = {}".format(
                    radius, torch.linalg.norm(y - temp), crit_B2, iter
                )
            )
        return sol



def sopt_fast_proj_B2(x, tau, params):
    r"""Fast projection algorithm onto the :math:`\ell_2`-ball.

    Compute the projection onto the :math:`\ell_2` ball, i.e. solve

    .. math::

        z^* = \min_{z} ||x - z||_2^2   s.t.  ||y - \Phi z||_2 < \tau

    where :math:`x` is the input vector and the solution :math:`z^*` is returned as sol.

    Args:
        x (np.ndarray): A sample position :math:`x` in the posterior space.

        tau (float): Radius of likelihood :math:`\ell_2`-ball.

        params (dict): Dictionary of parameters defining the optimisation.

    Returns:
        np.ndarray: Optimal solution :math:`z^*` of proximal projection.

    Notes:
        [1] M.J. Fadili and J-L. Starck, "Monotone operator splitting for optimization problems in sparse recovery" , IEEE ICIP, Cairo, Egypt, 2009.

        [2] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",  SIAM Journal on Imaging Sciences 2 (2009), no. 1, 183--202.
    """
    # Lambda function for scaling, used for tight frames only
    sc = lambda z: z*np.minimum(tau / np.linalg.norm(z), 1)

    # TIGHT FRAMES
    if (params["tight"]) and (params["pos"] or params["reality"]):

        temp = params["Phi"].dir_op(x) - params["y"]
        sol = x + 1 / params["nu"] * params["Phi"].adj_op(sc(temp) - temp)
        crit_B2 = "TOL_EPS"
        iter = 0
        u = 0

    # NON-TIGHT FRAMES
    else:

        # Initializations
        sol = x
        # u = params['u']
        u = params["Phi"].dir_op(sol)
        v = u
        iter = 1
        told = 1

        # Tolerance onto the L2 ball
        epsilon_low = tau / (1 + params["tol"])
        epsilon_up = tau / (1 - params["tol"])

        # Check if we are in the L2 ball
        dummy = params["Phi"].dir_op(sol)
        norm_res = np.linalg.norm(params["y"] - dummy, 2)
        if norm_res <= epsilon_up:
            crit_B2 = "TOL_EPS"
            true = 0

        # Projection onto the L2-ball
        if params["verbose"] > 1:
            print("  Proj. B2:")

        while 1:

            # Residual
            res = params["Phi"].dir_op(sol) - params["y"]
            norm_res = np.linalg.norm(res)

            # Scaling for the projection
            res = u * params["nu"] + res
            norm_proj = np.linalg.norm(res)

            # Log
            if params["verbose"] > 1:
                print(
                    "   Iter {}, epsilon = {}, ||y - Phi(x)||_2 = {}".format(
                        iter, tau, norm_res
                    )
                )

            # Stopping criterion
            if (norm_res >= epsilon_low) and (norm_res <= epsilon_up):
                crit_B2 = "TOL_EPS"
                break
            elif iter >= params["max_iter"]:
                crit_B2 = "MAX_IT"
                break

            # Projection onto the L2 ball
            t = (1 + np.sqrt(1 + 4 * told**2)) / 2
            ratio = np.minimum(1, tau / norm_proj)
            u = v
            v = 1 / params["nu"] * (res - res * ratio)
            u = v + (told - 1) / t * (v - u)

            # Current estimate
            sol = x - params["Phi"].adj_op(u)

            # Projection onto the non-negative orthant (positivity constraint)
            if params["pos"]:
                sol = np.real(sol)
                sol[sol < 0] = 0

            # Projection onto the real orthant (reality constraint)
            if params["reality"]:
                sol = np.real(sol)

            # Increment iteration labels
            told = t
            iter = iter + 1

    # Log after the projection onto the L2-ball
    if params["verbose"] >= 1:
        temp = params["Phi"].dir_op(sol)
        print(
            "  Proj. B2: epsilon = {}, ||y - Phi(x)||_2 = {}, {}, iter = {}".format(
                tau, np.linalg.norm(params["y"] - temp), crit_B2, iter
            )
        )

    return sol
