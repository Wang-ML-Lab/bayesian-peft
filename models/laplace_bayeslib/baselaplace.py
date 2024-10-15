from math import sqrt, pi, log
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.distributions import MultivariateNormal

from laplace_bayeslib.utils import (parameters_per_layer, invsqrt_precision, 
                           get_nll, validate, Kron, normal_samples)
from laplace_bayeslib.curvature import AsdlGGN, AsdlHessian
from tqdm import tqdm
import time
import random

__all__ = ['BaseLaplace', 'ParametricLaplace',
           'FullLaplace', 'KronLaplace', 'DiagLaplace', 'LowRankLaplace']



class BaseLaplace:
    """Baseclass for all Laplace approximations in this library.

    Parameters
    ----------
    model : torch.nn.Module
    likelihood : {'classification', 'regression'}
        determines the log likelihood Hessian approximation
    sigma_noise : torch.Tensor or float, default=1
        observation noise for the regression setting; must be 1 for classification
    prior_precision : torch.Tensor or float, default=1
        prior precision of a Gaussian prior (= weight decay);
        can be scalar, per-layer, or diagonal in the most general case
    prior_mean : torch.Tensor or float, default=0
        prior mean of a Gaussian prior, useful for continual learning
    temperature : float, default=1
        temperature of the likelihood; lower temperature leads to more
        concentrated posterior and vice versa.
    backend : subclasses of `laplace.curvature.CurvatureInterface`
        backend for access to curvature/Hessian approximations
    backend_kwargs : dict, default=None
        arguments passed to the backend on initialization, for example to
        set the number of MC samples for stochastic approximations.
    """
    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=None,
                 prior_mean=0., temperature=1., backend=None, backend_kwargs=None):
        if likelihood not in ['classification', 'regression']:
            raise ValueError(f'Invalid likelihood type {likelihood}')

        self.model = model
        self._device = next(model.parameters()).device
        print(self._device)

        # self.n_params = len(parameters_to_vector(self.model.parameters()).detach())
        # self.n_layers = len(list(self.model.parameters()))

        # Get a list of all trainable parameters
        trainable_parameters = []
        for name,p in self.model.named_parameters():
            if p.requires_grad and 'modules_to_save' not in name:
                trainable_parameters.append(p)
        # trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

        # Get the number of trainable parameters
        self.n_params = len(parameters_to_vector(trainable_parameters).detach())

        # # Get a list of all modules with at least one trainable parameter
        # trainable_layers = [m for m in self.model.modules() if any(p.requires_grad for p in m.parameters())]

        # Get the number of layers with trainable parameters
        # self.n_layers = len(trainable_layers)
        self.n_layers = len(trainable_parameters)

        if prior_precision is not None:
            self.prior_precision = prior_precision
        elif prior_precision is None:
            self.prior_precision = torch.ones(self.n_layers).to(self._device)      #prior_precision

        self.prior_mean = prior_mean
        if sigma_noise != 1 and likelihood != 'regression':
            raise ValueError('Sigma noise != 1 only available for regression.')
        self.likelihood = likelihood
        self.sigma_noise = sigma_noise
        self.temperature = temperature

        if backend is None:
            backend = AsdlGGN
        self._backend = None
        self._backend_cls = backend
        self._backend_kwargs = dict() if backend_kwargs is None else backend_kwargs

        # log likelihood = g(loss)
        self.loss = 0.
        self.n_outputs = None
        self.n_data = 0

    @property
    def backend(self):
        return self._backend_cls(self.model, self.likelihood,
                                              **self._backend_kwargs)

    def _curv_closure(self, batch, N):
        raise NotImplementedError

    def fit(self, train_loader):
        raise NotImplementedError

    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None):
        raise NotImplementedError

    @property
    def log_likelihood(self):
        """Compute log likelihood on the training data after `.fit()` has been called.
        The log likelihood is computed on-demand based on the loss and, for example,
        the observation noise which makes it differentiable in the latter for
        iterative updates.

        Returns
        -------
        log_likelihood : torch.Tensor
        """
        factor = - self._H_factor
        if self.likelihood == 'regression':
            # loss used is just MSE, need to add normalizer for gaussian likelihood
            c = self.n_data * self.n_outputs * torch.log(self.sigma_noise * sqrt(2 * pi))
            return factor * self.loss - c
        else:
            # for classification Xent == log Cat
            return factor * self.loss

    def __call__(self, x, pred_type, link_approx, n_samples):
        raise NotImplementedError

    def predictive(self, x, pred_type, link_approx, n_samples):
        return self(x, pred_type, link_approx, n_samples)

    def _check_jacobians(self, Js):
        if not isinstance(Js, torch.Tensor):
            raise ValueError('Jacobians have to be torch.Tensor.')
        if not Js.device == self._device:
            raise ValueError('Jacobians need to be on the same device as Laplace.')
        m, k, p = Js.size()
        if p != self.n_params:
            raise ValueError('Invalid Jacobians shape for Laplace posterior approx.')

    @property
    def prior_precision_diag(self):
        """Obtain the diagonal prior precision \\(p_0\\) constructed from either
        a scalar, layer-wise, or diagonal prior precision.

        Returns
        -------
        prior_precision_diag : torch.Tensor
        """
        if len(self.prior_precision) == 1:  # scalar
            return self.prior_precision * torch.ones(self.n_params, device=self._device)

        elif len(self.prior_precision) == self.n_params:  # diagonal
            return self.prior_precision

        elif len(self.prior_precision) == self.n_layers:  # per layer
            n_params_per_layer = parameters_per_layer(self.model)
            return torch.cat([prior * torch.ones(n_params, device=self._device) for prior, n_params
                              in zip(self.prior_precision, n_params_per_layer)])
        elif len(self.prior_precision) > 1 and len(self.prior_precision) < self.n_layers:
            n_params_per_layer = parameters_per_layer(self.model)
            num_last = len(self.prior_precision) - 1
            prior_prec_diag = []
            for n_params in n_params_per_layer[:-num_last]:
                prior_prec_diag.append(self.prior_precision[0] * torch.ones(n_params, device=self._device))
            for prior, n_params in zip(self.prior_precision[-num_last:], n_params_per_layer[-num_last:]):
                prior_prec_diag.append(prior * torch.ones(n_params, device=self._device))
            return torch.cat(prior_prec_diag)
        else:
            raise ValueError('Mismatch of prior and model. Diagonal, scalar, or per-layer prior.')

    @property
    def prior_mean(self):
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, prior_mean):
        if np.isscalar(prior_mean) and np.isreal(prior_mean):
            self._prior_mean = torch.tensor(prior_mean, device=self._device)
        elif torch.is_tensor(prior_mean):
            if prior_mean.ndim == 0:
                self._prior_mean = prior_mean.reshape(-1).to(self._device)
            elif prior_mean.ndim == 1:
                if not len(prior_mean) in [1, self.n_params]:
                    raise ValueError('Invalid length of prior mean.')
                self._prior_mean = prior_mean
            else:
                raise ValueError('Prior mean has too many dimensions!')
        else:
            raise ValueError('Invalid argument type of prior mean.')

    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision):
        self._posterior_scale = None
        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.tensor([prior_precision], device=self._device)
        elif torch.is_tensor(prior_precision):
            if prior_precision.ndim == 0:
                # make dimensional
                self._prior_precision = prior_precision.reshape(-1).to(self._device)
            elif prior_precision.ndim == 1:
                if len(prior_precision) not in [1, self.n_layers, self.n_params, 2, 3]:
                    raise ValueError('Length of prior precision does not align with architecture.')
                self._prior_precision = prior_precision.to(self._device)
            else:
                raise ValueError('Prior precision needs to be at most one-dimensional tensor.')
        else:
            raise ValueError('Prior precision either scalar or torch.Tensor up to 1-dim.')

    def optimize_prior_precision_base(self, pred_type, method='marglik', n_steps=100, lr=1e-1,
                                      init_prior_prec=1., val_loader=None, loss=get_nll,
                                      log_prior_prec_min=-2, log_prior_prec_max=10, grid_size=500,
                                      link_approx='probit', n_samples=100, verbose=False,
                                      cv_loss_with_var=False):
        """Optimize the prior precision post-hoc using the `method`
        specified by the user.

        Parameters
        ----------
        pred_type : {'glm', 'nn', 'gp'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive or Gaussian Process (GP) inference.
            The GLM predictive is consistent with the curvature approximations used here.
        method : {'marglik', 'CV'}, default='marglik'
            specifies how the prior precision should be optimized.
        n_steps : int, default=100
            the number of gradient descent steps to take.
        lr : float, default=1e-1
            the learning rate to use for gradient descent.
        init_prior_prec : float, default=1.0
            initial prior precision before the first optimization step.
        val_loader : torch.data.utils.DataLoader, default=None
            DataLoader for the validation set; each iterate is a training batch (X, y).
        loss : callable, default=get_nll
            loss function to use for CV.
        cv_loss_with_var: bool, default=False
            if true, `loss` takes three arguments `loss(output_mean, output_var, target)`,
            otherwise, `loss` takes two arguments `loss(output_mean, target)`
        log_prior_prec_min : float, default=-4
            lower bound of gridsearch interval for CV.
        log_prior_prec_max : float, default=4
            upper bound of gridsearch interval for CV.
        grid_size : int, default=100
            number of values to consider inside the gridsearch interval for CV.
        link_approx : {'mc', 'probit', 'bridge'}, default='probit'
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only `'mc'` is possible.
        n_samples : int, default=100
            number of samples for `link_approx='mc'`.
        verbose : bool, default=False
            if true, the optimized prior precision will be printed
            (can be a large tensor if the prior has a diagonal covariance).
        """

        if method == 'marglik':
            # self.prior_precision = init_prior_prec
            log_prior_prec = self.prior_precision.log()
            log_prior_prec.requires_grad = True
            optimizer = torch.optim.Adam([log_prior_prec], lr=lr)
            for _ in tqdm(range(n_steps)):
                optimizer.zero_grad()
                prior_prec = log_prior_prec.exp()
                neg_log_marglik = -self.log_marginal_likelihood(prior_precision=prior_prec)
                neg_log_marglik.backward()
                optimizer.step()
                if (_+1) % 100 == 0:
                    print(_, neg_log_marglik)
            self.prior_precision = log_prior_prec.detach().exp()
            del prior_prec, neg_log_marglik
            torch.cuda.empty_cache()


        elif method == 'CV':
            # replaces cross validation to batched gradient descent
            if val_loader is None:
                raise ValueError('CV requires a validation set DataLoader')
            interval = torch.logspace(
                log_prior_prec_min, log_prior_prec_max, grid_size
            )

            def divide_into_batches(data, batch_size):
                return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

            data_list = []
            for batch in tqdm(val_loader):
                try:
                    batch.to(self._device)
                except:
                    try: 
                        batch = {k: v.to(self._device) for k, v in batch.items()}
                    except:
                        prompts, _, _ = batch
                        prompts = prompts.to(self._device)
                        batch = prompts
                Js, f_mu = self.backend.jacobians(batch)
                for j, f, t in zip(Js, f_mu, batch['labels']):
                    data_list.append((j.detach().cpu(), f.detach().cpu(), t))

            batch_size = 8

            # self.prior_precision = init_prior_prec
            log_prior_prec = self.prior_precision.log()
            log_prior_prec.requires_grad = True
            optimizer = torch.optim.Adam([log_prior_prec], lr=lr)

            samples = 100000
            grad_step = 0
            while grad_step <= n_steps:
                nll_total = 0
                random.shuffle(data_list)
                shuffled_batches = divide_into_batches(data_list, batch_size)

                for batch in shuffled_batches:
                    Js, f_mu, target = zip(*batch)
                    Js = torch.stack(Js).to(self._device)
                    f_mu = torch.stack(f_mu).to(self._device)
                    target = torch.tensor(target).to(self._device)
                    
                    optimizer.zero_grad()
                    # self.prior_precision = torch.clamp(log_prior_prec.exp(), min=1e-4, max=1e5)
                    self.prior_precision = log_prior_prec.exp()

                    # print('optimising prior precision', Js.shape, f_mu.shape, target.shape, self.posterior_precision.__len__())

                    f_var = self.functional_variance(Js.to(self._device))
                    f_mu = f_mu.expand(samples, -1, -1).to(self._device)
                    f_var = f_var.expand(samples, -1, -1, -1)
                    eps = torch.randn_like(f_mu).unsqueeze(-1).to(f_mu.dtype).to(self._device)
                    probs = torch.softmax(f_mu + (torch.linalg.cholesky(f_var + torch.eye(f_var.shape[-1]).to(f_var.device)*1e-6).to(f_mu.dtype) @ eps).squeeze(-1), dim=-1).mean(0)
                    nll = -torch.log(probs[torch.arange(probs.shape[0]), target]).sum()
                    nll.backward()
                    optimizer.step()
                    nll_total += nll.detach().item()
                    grad_step += 1
                    if grad_step > n_steps:
                        break
                print(nll_total, log_prior_prec.exp().detach())

            self.prior_precision = log_prior_prec.detach().clone().exp()
            del data_list, shuffled_batches
        
        
    @property
    def sigma_noise(self):
        return self._sigma_noise

    @sigma_noise.setter
    def sigma_noise(self, sigma_noise):
        self._posterior_scale = None
        if np.isscalar(sigma_noise) and np.isreal(sigma_noise):
            self._sigma_noise = torch.tensor(sigma_noise, device=self._device)
        elif torch.is_tensor(sigma_noise):
            if sigma_noise.ndim == 0:
                self._sigma_noise = sigma_noise.to(self._device)
            elif sigma_noise.ndim == 1:
                if len(sigma_noise) > 1:
                    raise ValueError('Only homoscedastic output noise supported.')
                self._sigma_noise = sigma_noise[0].to(self._device)
            else:
                raise ValueError('Sigma noise needs to be scalar or 1-dimensional.')
        else:
            raise ValueError('Invalid type: sigma noise needs to be torch.Tensor or scalar.')

    @property
    def _H_factor(self):
        sigma2 = self.sigma_noise.square()
        return 1 / sigma2 / self.temperature


class ParametricLaplace(BaseLaplace):
    """
    Parametric Laplace class.

    Subclasses need to specify how the Hessian approximation is initialized,
    how to add up curvature over training data, how to sample from the
    Laplace approximation, and how to compute the functional variance.

    A Laplace approximation is represented by a MAP which is given by the
    `model` parameter and a posterior precision or covariance specifying
    a Gaussian distribution \\(\\mathcal{N}(\\theta_{MAP}, P^{-1})\\).
    The goal of this class is to compute the posterior precision \\(P\\)
    which sums as
    \\[
        P = \\sum_{n=1}^N \\nabla^2_\\theta \\log p(\\mathcal{D}_n \\mid \\theta)
        \\vert_{\\theta_{MAP}} + \\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}}.
    \\]
    Every subclass implements different approximations to the log likelihood Hessians,
    for example, a diagonal one. The prior is assumed to be Gaussian and therefore we have
    a simple form for \\(\\nabla^2_\\theta \\log p(\\theta) \\vert_{\\theta_{MAP}} = P_0 \\).
    In particular, we assume a scalar, layer-wise, or diagonal prior precision so that in
    all cases \\(P_0 = \\textrm{diag}(p_0)\\) and the structure of \\(p_0\\) can be varied.
    """

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=None,
                 prior_mean=0., temperature=1., backend=None, backend_kwargs=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, backend_kwargs)
        if not hasattr(self, 'H'):
            self._init_H()
            # posterior mean/mode
            self.mean = self.prior_mean

    def _init_H(self):
        raise NotImplementedError
    
    def _check_H_init(self):
        if self.H is None:
            raise AttributeError('Laplace not fitted. Run fit() first.')

    def fit(self, train_loader, override=True):
        """Fit the local Laplace approximation at the parameters of the model.

        Parameters
        ----------
        train_loader : torch.data.utils.DataLoader
            each iterate is a training batch (X, y);
            `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
        override : bool, default=True
            whether to initialize H, loss, and n_data again; setting to False is useful for
            online learning settings to accumulate a sequential posterior approximation.
        """
        if override:
            self._init_H()
            self.loss = 0
            self.n_data = 0

        self.model.eval()
        # self.mean = parameters_to_vector(self.model.parameters()).detach()
        mean = []
        for name, param in self.model.named_parameters():
            # print(name, param.shape, param.requires_grad)
            if param.requires_grad and 'modules_to_save' not in name:
                # print('appending')
                mean.append(param.detach())
        self.mean = parameters_to_vector(mean).detach()
        
        print('parameters shape', self.mean.shape)

        batch = next(iter(train_loader))
        try:
            batch = batch.to(self._device)
            with torch.no_grad():
                out = self.model(**batch)
        except:
            try: 
                batch = {k: v.to(self._device) for k, v in batch.items()}
                with torch.no_grad():
                    out = self.model(**batch)
            except:
                prompts, _, _ = batch
                inputs = prompts.to(self._device)
                with torch.no_grad():
                    out = self.model(**inputs)

        self.n_outputs = out.shape[-1]
        setattr(self.model, 'output_size', self.n_outputs)
        print('output shape', self.n_outputs)

        N = len(train_loader.dataset)

        for batch in tqdm(train_loader):
            try:
                batch = batch.to(self._device)
                with torch.no_grad():
                    out = self.model(**batch)
            except:
                try: 
                    batch = {k: v.to(self._device) for k, v in batch.items()}
                    with torch.no_grad():
                        out = self.model(**batch)
                except:
                    prompts, classes, _ = batch
                    batch = prompts.to(self._device)
                    with torch.no_grad():
                        out = self.model(**batch)
                    golds = classes.to(self._device)
                    batch.update({'labels': golds})

            self._backend = None
            self.model.zero_grad()

            loss_batch, H_batch,f = self._curv_closure(batch, N)
            self.loss += loss_batch
            self.H += H_batch

            del loss_batch, H_batch

        self.n_data += N

        print('H len', self.H.__len__())


    @property
    def scatter(self):
        """Computes the _scatter_, a term of the log marginal likelihood that
        corresponds to L-2 regularization:
        `scatter` = \\((\\theta_{MAP} - \\mu_0)^{T} P_0 (\\theta_{MAP} - \\mu_0) \\).

        Returns
        -------
        [type]
            [description]
        """
        delta = (self.mean - self.prior_mean).to(self.prior_precision_diag.dtype)
        # print('delta shape', delta.shape, self.prior_precision_diag.shape)
        # print('delta dtype', delta.dtype, self.prior_precision_diag.dtype)
        return (delta * self.prior_precision_diag) @ delta

    @property
    def log_det_prior_precision(self):
        """Compute log determinant of the prior precision
        \\(\\log \\det P_0\\)

        Returns
        -------
        log_det : torch.Tensor
        """
        return self.prior_precision_diag.log().sum()

    @property
    def log_det_posterior_precision(self):
        """Compute log determinant of the posterior precision
        \\(\\log \\det P\\) which depends on the subclasses structure
        used for the Hessian approximation.

        Returns
        -------
        log_det : torch.Tensor
        """
        raise NotImplementedError

    @property
    def log_det_ratio(self):
        """Compute the log determinant ratio, a part of the log marginal likelihood.
        \\[
            \\log \\frac{\\det P}{\\det P_0} = \\log \\det P - \\log \\det P_0
        \\]

        Returns
        -------
        log_det_ratio : torch.Tensor
        """
        return self.log_det_posterior_precision - self.log_det_prior_precision

    def square_norm(self, value):
        """Compute the square norm under post. Precision with `value-self.mean` as 𝛥:
        \\[
            \\Delta^\top P \\Delta
        \\]
        Returns
        -------
        square_form
        """
        raise NotImplementedError

    def log_prob(self, value, normalized=True):
        """Compute the log probability under the (current) Laplace approximation.

        Parameters
        ----------
        normalized : bool, default=True
            whether to return log of a properly normalized Gaussian or just the
            terms that depend on `value`.

        Returns
        -------
        log_prob : torch.Tensor
        """
        if not normalized:
            return - self.square_norm(value) / 2
        log_prob = - self.n_params / 2 * log(2 * pi) + self.log_det_posterior_precision / 2
        log_prob -= self.square_norm(value) / 2
        return log_prob

    def log_marginal_likelihood(self, prior_precision=None, sigma_noise=None):
        """Compute the Laplace approximation to the log marginal likelihood subject
        to specific Hessian approximations that subclasses implement.
        Requires that the Laplace approximation has been fit before.
        The resulting torch.Tensor is differentiable in `prior_precision` and
        `sigma_noise` if these have gradients enabled.
        By passing `prior_precision` or `sigma_noise`, the current value is
        overwritten. This is useful for iterating on the log marginal likelihood.

        Parameters
        ----------
        prior_precision : torch.Tensor, optional
            prior precision if should be changed from current `prior_precision` value
        sigma_noise : [type], optional
            observation noise standard deviation if should be changed

        Returns
        -------
        log_marglik : torch.Tensor
        """
        # update prior precision (useful when iterating on marglik)
        if prior_precision is not None:
            self.prior_precision = prior_precision

        # update sigma_noise (useful when iterating on marglik)
        if sigma_noise is not None:
            if self.likelihood != 'regression':
                raise ValueError('Can only change sigma_noise for regression.')
            self.sigma_noise = sigma_noise

        return self.log_likelihood - 0.5 * (self.log_det_ratio + self.scatter)

    def __call__(self, batch, pred_type='glm', link_approx='probit', n_samples=100, 
                 diagonal_output=False, generator=None):
        """Compute the posterior predictive on input data `x`.

        Parameters
        ----------
        x : torch.Tensor
            `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        link_approx : {'mc', 'probit', 'bridge', 'bridge_norm'}
            how to approximate the classification link function for the `'glm'`.
            For `pred_type='nn'`, only 'mc' is possible. 

        n_samples : int
            number of samples for `link_approx='mc'`.

        diagonal_output : bool
            whether to use a diagonalized posterior predictive on the outputs.
            Only works for `pred_type='glm'` and `link_approx='mc'`.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used)

        Returns
        -------
        predictive: torch.Tensor or Tuple[torch.Tensor]
            For `likelihood='classification'`, a torch.Tensor is returned with
            a distribution over classes (similar to a Softmax).
            For `likelihood='regression'`, a tuple of torch.Tensor is returned
            with the mean and the predictive variance.
        """
        f_mu = self.model(**batch)

        setattr(self.model, 'output_size', self.n_outputs)

        if pred_type not in ['glm', 'nn']:
            raise ValueError('Only glm and nn supported as prediction types.')

        if link_approx not in ['mc', 'probit', 'bridge', 'bridge_norm']:
            raise ValueError(f'Unsupported link approximation {link_approx}.')

        if pred_type == 'nn' and link_approx != 'mc':
            raise ValueError('Only mc link approximation is supported for nn prediction type.')
        
        if generator is not None:
            if not isinstance(generator, torch.Generator) or generator.device != batch['labels'].device:
                raise ValueError('Invalid random generator (check type and device).')

        if pred_type == 'glm':
            f_mu, f_var = self._glm_predictive_distribution(batch)
            # regression
            if self.likelihood == 'regression':
                return f_mu, f_var
            # classification
            if link_approx == 'mc':
                return self.predictive_samples(batch, pred_type='glm', n_samples=n_samples, 
                                               diagonal_output=diagonal_output).mean(dim=0)
            elif link_approx == 'probit':
                kappa = 1 / torch.sqrt(1. + np.pi / 8 * f_var.diagonal(dim1=1, dim2=2))
                # print(torch.sqrt(f_var.diagonal(dim1=1, dim2=2)))
                return torch.softmax(kappa * f_mu, dim=-1)
                # return torch.softmax(f_mu, dim=-1)
            elif 'bridge' in link_approx:
                # zero mean correction
                f_mu -= (f_var.sum(-1) * f_mu.sum(-1).reshape(-1, 1) /
                         f_var.sum(dim=(1, 2)).reshape(-1, 1))
                f_var -= (torch.einsum('bi,bj->bij', f_var.sum(-1), f_var.sum(-2)) /
                          f_var.sum(dim=(1, 2)).reshape(-1, 1, 1))
                # Laplace Bridge
                _, K = f_mu.size(0), f_mu.size(-1)
                f_var_diag = torch.diagonal(f_var, dim1=1, dim2=2)
                # optional: variance correction
                if link_approx == 'bridge_norm':
                    f_var_diag_mean = f_var_diag.mean(dim=1)
                    f_var_diag_mean /= torch.as_tensor([K/2], device=self._device).sqrt()
                    f_mu /= f_var_diag_mean.sqrt().unsqueeze(-1)
                    f_var_diag /= f_var_diag_mean.unsqueeze(-1)
                sum_exp = torch.exp(-f_mu).sum(dim=1).unsqueeze(-1)
                alpha = (1 - 2/K + f_mu.exp() / K**2 * sum_exp) / f_var_diag
                return torch.nan_to_num(alpha / alpha.sum(dim=1).unsqueeze(-1), nan=1.0)
        else:
            samples = self._nn_predictive_samples(x, n_samples)
            if self.likelihood == 'regression':
                return samples.mean(dim=0), samples.var(dim=0)
            return samples.mean(dim=0)

    def predictive_samples(self, x, pred_type='glm', n_samples=100, 
                           diagonal_output=False, generator=None):
        """Sample from the posterior predictive on input data `x`.
        Can be used, for example, for Thompson sampling.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch_size, input_shape)`

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        n_samples : int
            number of samples

        diagonal_output : bool
            whether to use a diagonalized glm posterior predictive on the outputs.
            Only applies when `pred_type='glm'`.

        generator : torch.Generator, optional
            random number generator to control the samples (if sampling used)

        Returns
        -------
        samples : torch.Tensor
            samples `(n_samples, batch_size, output_shape)`
        """
        if pred_type not in ['glm', 'nn']:
            raise ValueError('Only glm and nn supported as prediction types.')

        if pred_type == 'glm':
            f_mu, f_var = self._glm_predictive_distribution(x)
            assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
            if diagonal_output:
                f_var = torch.diagonal(f_var, dim1=1, dim2=2)
            f_samples = normal_samples(f_mu, f_var, n_samples, generator)
            if self.likelihood == 'regression':
                return f_samples
            return torch.softmax(f_samples, dim=-1)

        else:  # 'nn'
            return self._nn_predictive_samples(x, n_samples)

    @torch.enable_grad()
    def _glm_predictive_distribution(self, batch):
        Js, f_mu = self.backend.jacobians(batch)
        # print(Js, Js.shape)
        # print('jacobian shape', Js.shape)
        # print('f_mu shape', f_mu.shape)
        f_var = self.functional_variance(Js)
        return f_mu.detach(), f_var.detach()

    def _nn_predictive_samples(self, X, n_samples=100):
        fs = list()
        for sample in self.sample(n_samples):
            vector_to_parameters(sample, self.model.parameters())
            fs.append(self.model(X.to(self._device)).detach())
        vector_to_parameters(self.mean, self.model.parameters())
        fs = torch.stack(fs)
        if self.likelihood == 'classification':
            fs = torch.softmax(fs, dim=-1)
        return fs

    def functional_variance(self, Jacs):
        """Compute functional variance for the `'glm'` predictive:
        `f_var[i] = Jacs[i] @ P.inv() @ Jacs[i].T`, which is a output x output
        predictive covariance matrix.
        Mathematically, we have for a single Jacobian
        \\(\\mathcal{J} = \\nabla_\\theta f(x;\\theta)\\vert_{\\theta_{MAP}}\\)
        the output covariance matrix
        \\( \\mathcal{J} P^{-1} \\mathcal{J}^T \\).

        Parameters
        ----------
        Jacs : torch.Tensor
            Jacobians of model output wrt parameters
            `(batch, outputs, parameters)`

        Returns
        -------
        f_var : torch.Tensor
            output covariance `(batch, outputs, outputs)`
        """
        raise NotImplementedError

    def sample(self, n_samples=100):
        """Sample from the Laplace posterior approximation, i.e.,
        \\( \\theta \\sim \\mathcal{N}(\\theta_{MAP}, P^{-1})\\).

        Parameters
        ----------
        n_samples : int, default=100
            number of samples
        """
        raise NotImplementedError

    def optimize_prior_precision(self, method='marglik', pred_type='glm', n_steps=100, lr=1e-1,
                                 init_prior_prec=1., val_loader=None, loss=get_nll,
                                 log_prior_prec_min=-4, log_prior_prec_max=4, grid_size=100,
                                 link_approx='probit', n_samples=100, verbose=False,
                                 cv_loss_with_var=False):
        assert pred_type in ['glm', 'nn']
        self.optimize_prior_precision_base(pred_type, method, n_steps, lr,
                                           init_prior_prec, val_loader, loss,
                                           log_prior_prec_min, log_prior_prec_max,
                                           grid_size, link_approx, n_samples,
                                           verbose, cv_loss_with_var)
        return self.prior_precision

    @property
    def posterior_precision(self):
        """Compute or return the posterior precision \\(P\\).

        Returns
        -------
        posterior_prec : torch.Tensor
        """
        raise NotImplementedError


class KronLaplace(ParametricLaplace):
    """Laplace approximation with Kronecker factored log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have for each parameter group, e.g., torch.nn.Module,
    that \\P\\approx Q \\otimes H\\.
    See `BaseLaplace` for the full interface and see
    `laplace.utils.matrix.Kron` and `laplace.utils.matrix.KronDecomposed` for the structure of
    the Kronecker factors. `Kron` is used to aggregate factors by summing up and
    `KronDecomposed` is used to add the prior, a Hessian factor (e.g. temperature),
    and computing posterior covariances, marginal likelihood, etc.
    Damping can be enabled by setting `damping=True`.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('all', 'kron')

    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=None,
                 prior_mean=0., temperature=1., backend=None, damping=False,
                 **backend_kwargs):
        self.damping = damping
        self.H_facs = None
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, **backend_kwargs)

    def _init_H(self):
        print('======_init_H======')
        self.H = Kron.init_from_model(self.model, self._device)

    def _curv_closure(self, batch, N):
        return self.backend.kron(batch, N=N)

    @staticmethod
    def _rescale_factors(kron, factor):
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= factor
        return kron

    def fit(self, train_loader, override=True):
        if override:
            self.H_facs = None

        if self.H_facs is not None:
            n_data_old = self.n_data
            n_data_new = len(train_loader.dataset)
            self._init_H()  # re-init H non-decomposed
            # discount previous Kronecker factors to sum up properly together with new ones
            self.H_facs = self._rescale_factors(self.H_facs, n_data_old / (n_data_old + n_data_new))

        super().fit(train_loader, override=override)

        if self.H_facs is None:
            self.H_facs = self.H
        else:
            # discount new factors that were computed assuming N = n_data_new
            self.H = self._rescale_factors(self.H, n_data_new / (n_data_new + n_data_old))
            self.H_facs += self.H
        # Decompose to self.H for all required quantities but keep H_facs for further inference
        self.H = self.H_facs.decompose(damping=self.damping)
    
    def fit_temp(self, train_loader, override=True, steps=1000):
        if override:
            self.H_facs = None

        if self.H_facs is not None:
            n_data_old = self.n_data
            n_data_new = len(train_loader.dataset)
            self._init_H()  # re-init H non-decomposed
            # discount previous Kronecker factors to sum up properly together with new ones
            self.H_facs = self._rescale_factors(self.H_facs, n_data_old / (n_data_old + n_data_new))

        super().fit_temp(train_loader, override=override, steps=steps)

        if self.H_facs is None:
            self.H_facs = self.H
        else:
            # discount new factors that were computed assuming N = n_data_new
            self.H = self._rescale_factors(self.H, n_data_new / (n_data_new + n_data_old))
            self.H_facs += self.H
        # Decompose to self.H for all required quantities but keep H_facs for further inference
        self.H = self.H_facs.decompose(damping=self.damping)

    @property
    def posterior_precision(self):
        """Kronecker factored Posterior precision \\(P\\).

        Returns
        -------
        precision : `laplace.utils.matrix.KronDecomposed`
        """
        self._check_H_init()
        return self.H * self._H_factor + self.prior_precision

    @property
    def log_det_posterior_precision(self):
        if type(self.H) is Kron:  # Fall back to diag prior
            return self.prior_precision_diag.log().sum()
        return self.posterior_precision.logdet()

    def square_norm(self, value):
        delta = value - self.mean
        if type(self.H) is Kron:  # fall back to prior
            return (delta * self.prior_precision_diag) @ delta
        return delta @ self.posterior_precision.bmm(delta, exponent=1)

    def functional_variance(self, Js):
        return self.posterior_precision.inv_square_form(Js)

    def sample(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        samples = self.posterior_precision.bmm(samples, exponent=-0.5)
        return self.mean.reshape(1, self.n_params) + samples.reshape(n_samples, self.n_params)

    @BaseLaplace.prior_precision.setter
    def prior_precision(self, prior_precision):
        # Extend setter from Laplace to restrict prior precision structure.
        super(KronLaplace, type(self)).prior_precision.fset(self, prior_precision)
        if len(self.prior_precision) not in [1, self.n_layers, 2, 3]:
            raise ValueError('Prior precision for Kron either scalar or per-layer.')


class LowRankLaplace(ParametricLaplace):
    """Laplace approximation with low-rank log likelihood Hessian (approximation). 
    The low-rank matrix is represented by an eigendecomposition (vecs, values).
    Based on the chosen `backend`, either a true Hessian or, for example, GGN
    approximation could be used.
    The posterior precision is computed as
    \\( P = V diag(l) V^T + P_0.\\)
    To sample, compute the functional variance, and log determinant, algebraic tricks 
    are usedto reduce the costs of inversion to the that of a \\(K \times K\\) matrix
    if we have a rank of K.
    
    See `BaseLaplace` for the full interface.
    """
    _key = ('all', 'lowrank')
    def __init__(self, model, likelihood, sigma_noise=1, prior_precision=None, prior_mean=0, 
                 temperature=1, backend=AsdlHessian, backend_kwargs=None):
        super().__init__(model, likelihood, sigma_noise=sigma_noise, 
                         prior_precision=prior_precision, prior_mean=prior_mean, 
                         temperature=temperature, backend=backend, backend_kwargs=backend_kwargs)
    
    def _init_H(self):
        self.H = None

    @property
    def V(self):
        (U, l), prior_prec_diag = self.posterior_precision
        return U / prior_prec_diag.reshape(-1, 1)

    @property
    def Kinv(self):
        (U, l), _ = self.posterior_precision
        return torch.inverse(torch.diag(1 / l) + U.T @ self.V)

    def fit(self, train_loader, override=True):
        # override fit since output of eighessian not additive across batch
        if not override:
            # LowRankLA cannot be updated since eigenvalue representation not additive
            raise ValueError('LowRank LA does not support updating.')

        self.model.eval()
        self.mean = parameters_to_vector(self.model.parameters()).detach()

        X, _ = next(iter(train_loader))
        with torch.no_grad():
            try:
                out = self.model(X[:1].to(self._device))
            except (TypeError, AttributeError):
                out = self.model(X.to(self._device))
        self.n_outputs = out.shape[-1]
        print('n_output', self.n_outputs)
        setattr(self.model, 'output_size', self.n_outputs)

        eigenvectors, eigenvalues, loss = self.backend.eig_lowrank(train_loader)
        self.H = (eigenvectors, eigenvalues)
        self.loss = loss

        self.n_data = len(train_loader.dataset)

    @property
    def posterior_precision(self):
        """Return correctly scaled posterior precision that would be constructed
        as H[0] @ diag(H[1]) @ H[0].T + self.prior_precision_diag.

        Returns
        -------
        H : tuple(eigenvectors, eigenvalues)
            scaled self.H with temperature and loss factors.
        prior_precision_diag : torch.Tensor
            diagonal prior precision shape `parameters` to be added to H.
        """
        self._check_H_init()
        return (self.H[0], self._H_factor * self.H[1]), self.prior_precision_diag

    def functional_variance(self, Jacs):
        prior_var = torch.einsum('ncp,nkp->nck', Jacs / self.prior_precision_diag, Jacs)
        Jacs_V = torch.einsum('ncp,pl->ncl', Jacs, self.V)
        info_gain = torch.einsum('ncl,nkl->nck', Jacs_V @ self.Kinv, Jacs_V)
        return prior_var - info_gain

    def sample(self, n_samples):
        samples = torch.randn(self.n_params, n_samples)
        d = self.prior_precision_diag
        Vs = self.V * d.sqrt().reshape(-1, 1)
        VtV = Vs.T @ Vs
        Ik = torch.eye(len(VtV))
        A = torch.linalg.cholesky(VtV)
        B = torch.linalg.cholesky(VtV + Ik)
        A_inv = torch.inverse(A)
        C = torch.inverse(A_inv.T @ (B - Ik) @ A_inv)
        Kern_inv = torch.inverse(torch.inverse(C) + Vs.T @ Vs)
        dinv_sqrt = (d).sqrt().reshape(-1, 1)
        prior_sample = dinv_sqrt * samples
        gain_sample = dinv_sqrt * Vs @ Kern_inv @ (Vs.T @ samples)
        return self.mean + (prior_sample - gain_sample).T

    @property
    def log_det_posterior_precision(self):
        (U, l), prior_prec_diag = self.posterior_precision
        return l.log().sum() + prior_prec_diag.log().sum() - torch.logdet(self.Kinv)


class DiagLaplace(ParametricLaplace):
    """Laplace approximation with diagonal log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have \\(P \\approx \\textrm{diag}(P)\\).
    See `BaseLaplace` for the full interface.
    """
    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ('all', 'diag')

    def _init_H(self):
        self.H = torch.zeros(self.n_params, device=self._device)

    def _curv_closure(self, batch, N):
        return self.backend.diag(batch, N=N)

    @property
    def posterior_precision(self):
        """Diagonal posterior precision \\(p\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        self._check_H_init()
        return self._H_factor * self.H + self.prior_precision_diag

    @property
    def posterior_scale(self):
        """Diagonal posterior scale \\(\\sqrt{p^{-1}}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision.sqrt()

    @property
    def posterior_variance(self):
        """Diagonal posterior variance \\(p^{-1}\\).

        Returns
        -------
        precision : torch.tensor
            `(parameters)`
        """
        return 1 / self.posterior_precision

    @property
    def log_det_posterior_precision(self):
        return self.posterior_precision.log().sum()

    def square_norm(self, value):
        delta = value - self.mean
        return delta @ (delta * self.posterior_precision)

    def functional_variance(self, Js: torch.Tensor) -> torch.Tensor:
        self._check_jacobians(Js)
        return torch.einsum('ncp,p,nkp->nck', Js, self.posterior_variance, Js)

    def sample(self, n_samples=100):
        samples = torch.randn(n_samples, self.n_params, device=self._device)
        samples = samples * self.posterior_scale.reshape(1, self.n_params)
        return self.mean.reshape(1, self.n_params) + samples


class FunctionalLaplace(BaseLaplace):
    pass


class SoDLaplace(FunctionalLaplace):
    pass
