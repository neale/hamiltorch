import torch
import torch.nn as nn
from enum import Enum

from numpy import pi
from . import util


class Metric(Enum):
    HESSIAN = 1
    SOFTABS = 2
    JACOBIAN_DIAG = 3

def collect_gradients(log_prob, params):
    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    else:
        params.grad = torch.autograd.grad(log_prob,params)[0]
    return params


def gibbs(
    params,
    log_prob_func=None,
    jitter=None,
    normalizing_const=1.,
    softabs_const=None,
    mass=None,
    metric=Metric.HESSIAN):

    if mass is None:
        dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    else:
        if len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), mass)
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass)
    return dist.sample()


def leapfrog(
        params,
        momentum,
        log_prob_func,
        steps=10,
        step_size=0.1,
        jitter=0.01,
        normalizing_const=1.,
        softabs_const=1e6,
        explicit_binding_const=100,
        fixed_point_threshold=1e-20,
        fixed_point_max_iterations=6,
        jitter_max_tries=10,
        inv_mass=None,
        ham_func=None,
        metric=Metric.HESSIAN):

    def params_grad(p):
        p = p.detach().requires_grad_()
        log_prob = log_prob_func(p)
        p = collect_gradients(log_prob, p)
        return p.grad
    ret_params = []
    ret_momenta = []
    momentum += 0.5 * step_size * params_grad(params)
    for n in range(steps):
        if inv_mass is None:
            params = params + step_size * momentum
        else:
            #Assume G is diag here so 1/Mass = G inverse
            if len(inv_mass.shape) == 2:
                params = params + step_size * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
            else:
                params = params + step_size * inv_mass * momentum
        p_grad = params_grad(params)
        momentum += step_size * p_grad
        ret_params.append(params.clone())
        ret_momenta.append(momentum.clone())
    # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
    ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
    return ret_params, ret_momenta


def acceptance(h_old, h_new):
    return float(-h_new + h_old)

def adaptation(rho, t, step_size_init, H_t, eps_bar, desired_accept_rate=0.8):
    # rho is current acceptance ratio
    # t is current iteration
    t = t + 1
    if util.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0 # Acceptance rate is zero if nan.
    else:
        alpha = min(1.,float(torch.exp(torch.FloatTensor([rho]))))
    mu = float(torch.log(10*torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1-(1/(t+t0)))*H_t + (1/(t+t0))*(desired_accept_rate - alpha)
    x_new = mu - (t**0.5)/gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t**-kappa * x_new +  (1 - t**-kappa) * torch.log(torch.FloatTensor([eps_bar]))
    eps_bar = float(torch.exp(x_new_bar))
    return step_size, eps_bar, H_t


def hamiltonian(
        params,
        momentum,
        log_prob_func,
        jitter=0.01,
        normalizing_const=1.,
        softabs_const=1e6,
        explicit_binding_const=100,
        inv_mass=None,
        ham_func=None,
        metric=Metric.HESSIAN):

    log_prob = log_prob_func(params)
    if util.has_nan_or_inf(log_prob):
        print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
        raise util.LogProbError()

    potential = -log_prob
    if inv_mass is None:
        kinetic = 0.5 * torch.dot(momentum, momentum)
    else:
        if len(inv_mass.shape) == 2:
            kinetic = 0.5 * torch.matmul(momentum.view(1,-1),torch.matmul(inv_mass,momentum.view(-1,1))).view(-1)
        else:
            kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)
    hamiltonian = potential + kinetic
    return hamiltonian


def sample(
        log_prob_func,
        params_init,
        num_samples=10,
        num_steps_per_sample=10,
        step_size=0.1,
        burn=0,
        jitter=None,
        inv_mass=None,
        normalizing_const=1.,
        softabs_const=None,
        explicit_binding_const=100,
        fixed_point_threshold=1e-5,
        fixed_point_max_iterations=1000,
        jitter_max_tries=10,
        metric=Metric.HESSIAN,
        desired_accept_rate=0.8):

    if params_init.dim() != 1:
        raise RuntimeError('params_init must be a 1d tensor.')

    if burn >= num_samples:
        raise RuntimeError('burn must be less than num_samples.')

    # Invert mass matrix once (As mass is used in Gibbs resampling step)
    mass = None
    if inv_mass is not None:
        if len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1/inv_mass

    params = params_init.clone().requires_grad_()
    ret_params = [params.clone()]
    num_rejected = 0
    util.progress_bar_init('Sampling (HMC; IMPLICIT)', num_samples, 'Samples')
    for n in range(num_samples):
        util.progress_bar_update(n)
        try:
            momentum = gibbs(params,
                log_prob_func=log_prob_func,
                jitter=jitter,
                normalizing_const=normalizing_const,
                softabs_const=softabs_const,
                metric=metric,
                mass=mass)

            ham = hamiltonian(params,
                momentum,
                log_prob_func,
                jitter=jitter,
                softabs_const=softabs_const,
                explicit_binding_const=explicit_binding_const,
                normalizing_const=normalizing_const,
                metric=metric,
                inv_mass=inv_mass)

            leapfrog_params, leapfrog_momenta = leapfrog(params,
                momentum,
                log_prob_func,
                steps=num_steps_per_sample,
                step_size=step_size,
                inv_mass=inv_mass,
                jitter=jitter,
                jitter_max_tries=jitter_max_tries,
                fixed_point_threshold=fixed_point_threshold,
                fixed_point_max_iterations=fixed_point_max_iterations,
                softabs_const=softabs_const,
                explicit_binding_const=explicit_binding_const,
                metric=metric)

            params = leapfrog_params[-1].detach().requires_grad_()
            momentum = leapfrog_momenta[-1]
            new_ham = hamiltonian(params,
                momentum,
                log_prob_func,
                jitter=jitter,
                softabs_const=softabs_const,
                explicit_binding_const=explicit_binding_const,
                normalizing_const=normalizing_const,
                metric=metric,
                inv_mass=inv_mass)

            rho = min(0., acceptance(ham, new_ham))

            if rho >= torch.log(torch.rand(1)):
                if n > burn:
                    ret_params.extend(leapfrog_params)
            else:
                num_rejected += 1
                params = ret_params[-1]
                if n > burn:
                    leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                    ret_params.extend(leapfrog_params) # append the current sample to the chain

        except util.LogProbError:
            num_rejected += 1
            params = ret_params[-1]
            if n > burn:
                leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                ret_params.extend(leapfrog_params)

    util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected/num_samples)) #need to adapt for burn
    return list(map(lambda t: t.detach(), ret_params))

