"""Extrapolator class"""

import torch

from torch.optim.optimizer import Optimizer


class Extrapolator(Optimizer):
    r"""Implements a BDF weight extrapolator algorithm.

    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        eta: learning rate (default: coming from optimizer)
        h: finite difference (default: eta)
        dt: extrapolation step size (default: 1e-3)

    Example:
        extrapolator = Extrapolator(model.parameters(), eta=lr, h=lr, dt=0.01
        for epoch in range(...):
            optimizer.zero_grad()
            feedforward(...)
            loss(...)
            optimizer.step()
            extrapolator.step()
    """

    def __init__(self, params, eta, h, dt):
        defaults = dict(eta=eta, h=h, dt=dt)
        super(Extrapolator, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Extrapolator, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:

            eta = group["eta"]
            h = group["h"]
            dt = group["dt"]

            for p in group['params']:

                if p.grad is None:
                    continue

                # Get gradients of parameters p
                d_p = p.grad.data

                # Buffer gradients, initially same gradients for all states
                param_state = self.state[p]

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    param_state['grad_1'] = torch.clone(d_p).detach()
                    param_state['grad_2'] = torch.clone(d_p).detach()
                else:
                    # if (state["step"]+1) % 6 == 0:
                    if state["step"] > 2:
                        grad_1 = param_state['grad_1']
                        grad_2 = param_state['grad_2']

                        # Extrapolation step
                        # First order part of extrapolation step
                        grad = 11.0*d_p - 7.0*grad_1 + 2.0*grad_2
                        #torch.nn.utils.clip_grad_norm_(grad, 1.0)
                        alpha = -((dt * eta) / (6.0 * h))
                        p.data.add_(grad, alpha=alpha)

                        # Second order part of extrapolation step
                        grad = 2.0*d_p - 3.0*grad_1 + grad_2
                        # torch.nn.utils.clip_grad_norm_(grad, 1.0)
                        alpha = -((dt**2 * eta) / (2.0 * h**2))
                        p.data.add_(grad, alpha=alpha)

                        # # Third order part of extrapolation step
                        grad = d_p - 2.0*grad_1 + grad_2
                        # torch.nn.utils.clip_grad_norm_(grad, 1.0)
                        alpha = -((dt**3 * eta) / (6.0 * h**3))
                        p.data.add_(grad, alpha=alpha)

                # First in, first out gradient buffer
                param_state['grad_2'] = param_state['grad_1']
                param_state['grad_1'] = torch.clone(d_p).detach()

                state["step"] += 1

        return loss

    def set_eta(self):
        state_dict = self.state_dict()
        for param_group in state_dict["param_groups"]:
            param_group["eta"] *= 0.1
        self.load_state_dict(state_dict)
