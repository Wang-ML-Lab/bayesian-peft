import warnings
import numpy as np
import torch

from laplace_bayeslib.curvature.asdl_lib.matrices import (
    FISHER_EXACT, FISHER_MC, FISHER_EMP, SHAPE_KRON, SHAPE_DIAG, SHAPE_FULL
)
from laplace_bayeslib.curvature.asdl_lib.grad_maker import LOSS_MSE, LOSS_CROSS_ENTROPY
from laplace_bayeslib.curvature.asdl_lib.fisher import FisherConfig, get_fisher_maker
# from asdl.fisher import calculate_fisher
from laplace_bayeslib.curvature.asdl_lib.hessian import HessianMaker, HessianConfig
from laplace_bayeslib.curvature.asdl_lib.gradient import batch_gradient

# from asdl import FISHER_EXACT, FISHER_MC #, COV
# from asdl import SHAPE_KRON, SHAPE_DIAG, SHAPE_FULL
# from asdl import fisher_for_cross_entropy
# from asdl.hessian import hessian_eigenvalues, hessian_for_loss
# from asdl.gradient import batch_gradient

import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from laplace_bayeslib.curvature.asdl_lib.core import extend
from laplace_bayeslib.curvature.asdl_lib.operations import OP_BATCH_GRADS
import laplace_bayeslib.curvature.asdl_lib as asdl_lib

from laplace_bayeslib.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace_bayeslib.utils import Kron, _is_batchnorm

def batch_gradient(model, closure, input_shape,return_outputs=False):
    with extend(model, OP_BATCH_GRADS) as cxt:
        outputs = closure()
        grads = []
        N = input_shape[0]
        L = input_shape[-1]
        for module in model.modules():
            g = cxt.batch_grads(module, flatten=True)
            if g is not None:
                if len(input_shape) == 2:
                    if g.shape[0] > N:
                        grads.append(g.reshape(*input_shape,-1).sum(-2))
                    else:
                        grads.append(g)
                else:
                    if g.shape[0] > N*L:
                        grads.append(g.reshape(*input_shape,-1).sum(-2).sum(-2))
                    elif g.shape[0] > N:
                        grads.append(g.reshape(*input_shape[:-1],-1).sum(-2))
                    else:
                        grads.append(g)
        grads = torch.cat(grads, dim=-1)  # (n, p)
    if return_outputs:
        return grads, outputs
    else:
        return grads
    
asdl_lib.batch_gradient = batch_gradient
    



EPS = 1e-6


class AsdlInterface(CurvatureInterface):
    """Interface for asdfghjkl backend.
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None,
                 kfac_conv='kfac-expand'):
        super().__init__(model, likelihood, last_layer, subnetwork_indices)
        self.kfac_conv = kfac_conv

    @property
    def loss_type(self):
        return LOSS_MSE if self.likelihood == 'regression' else LOSS_CROSS_ENTROPY

    def jacobians(self, batch):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        x = batch['input_ids']
        input_shape =  x.shape

        Js = list()
        for i in range(self.model.output_size):
            def closure():
                self.model.zero_grad()
                f = self.model(**batch)
                loss = f[:, i].sum()
                loss.backward()
                return f

            Ji, f = batch_gradient(self.model, closure, x.shape, return_outputs=True)
            if self.subnetwork_indices is not None:
                Ji = Ji[:, self.subnetwork_indices]
            # if Ji.shape[0] > N:
                # p = Ji.shape[-1]
                # Ji = Ji.reshape(N,L,p).sum(1)
            Js.append(Ji)
        Js = torch.stack(Js, dim=1)
        return Js, f

    def gradients(self, x, y):
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter
        \\(\\theta\\) using asdfghjkl's backend.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        y : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
        Gs : torch.Tensor
            gradients `(batch, parameters)`
        """
        print('=======using gradients() function=======')
        def closure():
            self.model.zero_grad()
            loss = self.lossfunc(self.model(x), y)
            loss.backward()
            return loss

        Gs, loss = batch_gradient(self.model, closure, return_outputs=True)
        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]
        return Gs, loss

    @property
    def _ggn_type(self):
        raise NotImplementedError

    def _get_kron_factors(self, M):
        kfacs = list()
        for module in self.model.modules():
            if _is_batchnorm(module):
                warnings.warn('BatchNorm unsupported for Kron, ignore.')
                continue

            stats = getattr(module, 'fisher', None)
            if stats is None:
                continue

            if hasattr(module, 'bias') and module.bias is not None:
                # split up bias and weights
                kfacs.append([stats.kron.B, stats.kron.A])
                kfacs.append([stats.kron.B])
            elif hasattr(module, 'weight'):
                p, q = np.prod(stats.kron.B.shape), np.prod(stats.kron.A.shape)
                if p == q == 1:
                    kfacs.append([stats.kron.B * stats.kron.A])
                else:
                    kfacs.append([stats.kron.B, stats.kron.A])
            else:
                raise ValueError(f'Whats happening with {module}?')
        return Kron(kfacs)

    @staticmethod
    def _rescale_kron_factors(kron, N):
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= 1/N
        return kron

    def diag(self, batch, **kwargs):
        y = batch['labels']
        batch.pop('labels')
        if self.last_layer:
            X = self.model.forward_with_features(**batch)

            cfg = FisherConfig(fisher_type=self._ggn_type, loss_type=self.loss_type,
                            fisher_shapes=[SHAPE_DIAG], data_size=1)
            fisher_maker = get_fisher_maker(self.model, cfg)
            if 'emp' in self._ggn_type:
                dummy = fisher_maker.setup_model_call(self._model, X)
                fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
            else:
                fisher_maker.setup_model_call(self._model, X)
        else:
            cfg = FisherConfig(fisher_type=self._ggn_type, loss_type=self.loss_type,
                            fisher_shapes=[SHAPE_DIAG], data_size=1)
            fisher_maker = get_fisher_maker(self.model, cfg)
            if 'emp' in self._ggn_type:
                dummy = fisher_maker.setup_model_call(self._model, **batch)
                fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
            else:
                fisher_maker.setup_model_call(self._model, **batch)

        f, _ = fisher_maker.forward_and_backward()
        print('f shape', f.shape)
        print('y shape', y.shape)
        loss = self.lossfunc(f.detach(), y)
        vec = list()
        for module in self.model.modules():
            stats = getattr(module, 'fisher', None)
            if stats is None:
                # print('stats is None', module)
                continue
            vec.extend(stats.to_vector())
        diag_ggn = torch.cat(vec)
        if self.subnetwork_indices is not None:
            diag_ggn = diag_ggn[self.subnetwork_indices]
        if type(self) is AsdlEF and self.likelihood == 'regression':
            curv_factor = 0.5  # correct scaling for diag ef
        else:
            curv_factor = 1.0   # ASDL uses proper 1/2 * MSELoss
        return self.factor * loss, curv_factor * diag_ggn, f.detach()
 
    def kron(self, batch, N, **kwargs):
        y = batch['labels']
        batch.pop('labels')
        if self.last_layer:
            X = self.model.forward_with_features(**batch)

            cfg = FisherConfig(fisher_type=self._ggn_type, loss_type=self.loss_type,
                           fisher_shapes=[SHAPE_KRON], data_size=1)
            # fisher_maker = get_fisher_maker(self.model, cfg, self.kfac_conv)
            fisher_maker = get_fisher_maker(self.model, cfg)
            if 'emp' in self._ggn_type:
                dummy = fisher_maker.setup_model_call(self._model, X)
                fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
            else:
                fisher_maker.setup_model_call(self._model, X)

        else:
            cfg = FisherConfig(fisher_type=self._ggn_type, loss_type=self.loss_type,
                            fisher_shapes=[SHAPE_KRON], data_size=1)
            # fisher_maker = get_fisher_maker(self.model, cfg, self.kfac_conv)
            fisher_maker = get_fisher_maker(self.model, cfg)
            if 'emp' in self._ggn_type:
                dummy = fisher_maker.setup_model_call(self._model, **batch)
                fisher_maker.setup_loss_call(self.lossfunc, dummy, y)
            else:
                fisher_maker.setup_model_call(self._model, **batch)

        f, _ = fisher_maker.forward_and_backward()
        loss = self.lossfunc(f.detach(), y)
        M = len(y)
        kron = self._get_kron_factors(M)
        kron = self._rescale_kron_factors(kron, N)
        if type(self) is AsdlEF and self.likelihood == 'regression':
            curv_factor = 0.5  # correct scaling for diag ef
        else:
            curv_factor = 1.0   # ASDL uses proper 1/2 * MSELoss
        return self.factor * loss, curv_factor * kron, f.detach()


class AsdlHessian(AsdlInterface):

    def __init__(self, model, likelihood, last_layer=False, low_rank=10):
        super().__init__(model, likelihood, last_layer)
        self.low_rank = low_rank

    @property
    def _ggn_type(self):
        raise NotImplementedError()

    def full(self, x, y, **kwargs):
        print(x)
        exit()
        if self.last_layer:
            x = self.model.forward_with_features(x)
        cfg = HessianConfig(hessian_shapes=[SHAPE_FULL])
        hess_maker = HessianMaker(self.model, cfg)
        dummy = hess_maker.setup_model_call(self._model, x)
        hess_maker.setup_loss_call(self.lossfunc, dummy, y)
        hess_maker.forward_and_backward()
        H = self._model.hessian.data
        loss = self.lossfunc(self.model(x), y).detach()
        return self.factor * loss, self.factor * H

    def eig_lowrank(self, data_loader):
        # TODO: need to implement manually...
        # compute truncated eigendecomposition of the Hessian, only keep eigvals > EPS
        if self.last_layer:
            x.pop('labels')
            x = self.model.forward_with_features(x)
        cfg = HessianConfig(hessian_shapes=[SHAPE_FULL])
        hess_maker = HessianMaker(self.model, cfg)
        dummy = hess_maker.setup_model_call(self._model, x)
        hess_maker.setup_loss_call(self.lossfunc, dummy, y)
        # iteratively go through data loader and average eigendecomposition
        # previously:
        eigvals, eigvecs = hessian_eig(self.model, self.lossfunc, data_loader=data_loader,
                                       top_n=self.low_rank, max_iters=self.low_rank*10)
        eigvals = torch.from_numpy(np.array(eigvals))
        mask = (eigvals > EPS)
        eigvecs = torch.stack([vec.get_flatten_vector() for vec in eigvecs], dim=1)[:, mask]
        device = eigvecs.device
        eigvals = eigvals[mask].to(eigvecs.dtype).to(device)
        loss = sum([self.lossfunc(self.model(x.to(device)).detach(), y.to(device)) for x, y in data_loader])
        return eigvecs, self.factor * eigvals, self.factor * loss


class AsdlGGN(AsdlInterface, GGNInterface):
    """Implementation of the `GGNInterface` using asdfghjkl.
    """
    def __init__(self, model, likelihood, last_layer=False, subnetwork_indices=None, stochastic=False,
                 kfac_conv='kfac-expand'):
        super().__init__(model, likelihood, last_layer, subnetwork_indices, kfac_conv=kfac_conv)
        self.stochastic = stochastic

    @property
    def _ggn_type(self):
        return FISHER_MC if self.stochastic else FISHER_EXACT


class AsdlEF(AsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using asdfghjkl.
    """
    def __init__(self, model, likelihood, last_layer=False, kfac_conv='kfac-expand'):
        super().__init__(model, likelihood, last_layer, kfac_conv=kfac_conv)

    @property
    def _ggn_type(self):
        return FISHER_EMP
