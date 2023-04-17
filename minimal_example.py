#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   minimal_example.py
@Time    :   2023/04/17 17:11:18
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""

import argparse
import logging
from typing import Dict

import pyro
import pyro.distributions as pyro_dist
import torch
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.ops.indexing import Vindex
from pyro.optim import ClippedAdam
from torch.nn.utils.rnn import pad_sequence


class MinimalExample:
    def __init__(self, hyper_params: Dict, data_dims: Dict, obs_data: Dict, obs_data_size: Dict,
                 arg_parser: argparse.ArgumentParser) -> None:
        logging.basicConfig(format="[%(asctime)s %(levelname)s]: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)
        self._hyper_params = hyper_params
        self._args = arg_parser.parse_args()
        self._device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self._data_dims = self._move_to_device(data_dims)
        self._obs_data = self._move_to_device(obs_data)
        self._obs_data_size = self._move_to_device(obs_data_size)
        self._max_plate_nesting = 2

    def _move_to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        elif isinstance(data, dict):
            return {key: self._move_to_device(value) for key, value in data.items()}
        else:
            return data

    def _model(self):
        phi = {}
        for attr_name in self._data_dims['DiscreteAttribute'].keys():
            attr_dim = self._data_dims['DiscreteAttribute'][attr_name]
            eta_a = F.softmax(torch.randint(1, 10, (attr_dim, ), device=self._device, dtype=torch.float32), dim=-1)
            phi[f'{attr_name}'] = pyro.sample(f'phi_{attr_name}',
                                              pyro_dist.Dirichlet(eta_a).expand(
                                                  torch.Size((self._hyper_params['M'], ))).to_event(1))  # (M, Dim_a)
        tau_1 = torch.tensor(5., device=self._device)
        tau_2 = torch.tensor(1., device=self._device)
        mu_gamma = torch.tensor(.5, device=self._device)
        sigma_gamma = torch.tensor(.5, device=self._device)
        xi_1 = pyro.sample('xi_1',
                           pyro_dist.Gamma(tau_1, tau_2).expand(torch.Size(
                               (self._hyper_params['M'], ))).to_event(1))  # (M, )
        xi_2 = pyro.sample('xi_2',
                           pyro_dist.Gamma(tau_1, tau_2).expand(torch.Size(
                               (self._hyper_params['M'], ))).to_event(1))  # (M, )
        gamma = pyro.sample('gamma',
                            pyro_dist.Normal(mu_gamma, sigma_gamma).expand(
                                torch.Size((self._hyper_params['M'],
                                            self._data_dims['DemographicInfo']))).to_event(2))  # (M, Dim_x)

        mu_omega = torch.tensor(0., device=self._device)
        sigma_omega = torch.tensor(.5, device=self._device)
        omega_b = pyro.sample('omega_browsed',
                              pyro_dist.Normal(mu_omega, sigma_omega).expand(
                                  torch.Size(
                                      (self._hyper_params['S'], self._hyper_params['M']))).to_event(2))  # (S, M)
        omega_c = pyro.sample('omega_clicked',
                              pyro_dist.Normal(mu_omega, sigma_omega).expand(
                                  torch.Size(
                                      (self._hyper_params['S'], self._hyper_params['M']))).to_event(2))  # (S, M)

        eta_s = 0.8 * torch.eye(self._hyper_params['S'], device=self._device) + 0.2
        with pyro.plate('users', self._data_dims['User'], dim=-1, device=self._device):
            p = pyro.sample('p', pyro_dist.Dirichlet(eta_s).to_event(1))

        max_session_length = self._data_dims['Session'].max()
        with pyro.plate('batched_users', self._data_dims['User'], self._args.batch_size, dim=-1,
                        device=self._device) as batch:
            session_length = self._data_dims['Session'][batch]
            state = 0
            x = self._obs_data_size['DemographicInfo'][batch]  # (Batch, Dim_x)
            for t in pyro.markov(range(max_session_length)):
                with poutine.mask(mask=t < session_length):
                    p_batch_state = p[batch, state, :]
                    state = pyro.sample(f'state_{t}',
                                        pyro_dist.Categorical(p_batch_state),
                                        infer={"enumerate": "parallel"})
                    omega_b_state = Vindex(omega_b)[state, :]
                    omega_c_state = Vindex(omega_c)[state, :]
                    zeta_b_t = ((gamma @ x.T).T + omega_b_state).exp()  # (Batch, M)
                    zeta_c_t = ((gamma @ x.T).T + omega_c_state).exp()  # (Batch, M)
                    theta_b_t = pyro.sample(f'theta_b_{t}', pyro_dist.Dirichlet(zeta_b_t))  # (Batch, M)
                    theta_c_t = pyro.sample(f'theta_c_{t}', pyro_dist.Dirichlet(zeta_c_t))  # (Batch, M)
                    browsed_cards_t = self._obs_data_size['Browsed'][t, batch]  # (Batch, )
                    max_browsed_cards_num = self._obs_data_size['Browsed'].max()
                    with pyro.plate(f'browsed_card_{t}', max_browsed_cards_num, dim=-2,
                                    device=self._device) as browsed_cards_plate:
                        with poutine.mask(mask=browsed_cards_plate.unsqueeze(-1) < browsed_cards_t.unsqueeze(0)):
                            z_b_t = pyro.sample(f'z_b_{t}',
                                                pyro_dist.Categorical(theta_b_t),
                                                infer={"enumerate": "parallel"})
                            for attr_name in self._data_dims['DiscreteAttribute'].keys():
                                p_z_b_t = Vindex(phi[f'{attr_name}'])[z_b_t]
                                if attr_name == 'talkId':
                                    for j in range(self._data_dims['talkId_num']):
                                        obs_data = self._obs_data['Obs_attrs_browsed'][attr_name][j, :, t, batch]
                                        d_b_t_a = pyro.sample(f'd_b_{t}_{attr_name}_{j}',
                                                              pyro_dist.Categorical(p_z_b_t),
                                                              obs=obs_data)  # obs
                                else:
                                    obs_data = self._obs_data['Obs_attrs_browsed'][attr_name][:, t, batch]
                                    d_b_t_a = pyro.sample(f'd_b_{t}_{attr_name}',
                                                          pyro_dist.Categorical(p_z_b_t),
                                                          obs=obs_data)  # obs

                    clicked_cards_t = self._obs_data_size['Clicked'][t, batch]
                    max_clicked_cards_num = self._obs_data_size['Clicked'].max()
                    with pyro.plate(f'clicked_cards_{t}', max_clicked_cards_num, dim=-2,
                                    device=self._device) as clicked_cards_plate:
                        with poutine.mask(mask=clicked_cards_plate.unsqueeze(-1) < clicked_cards_t.unsqueeze(0)):
                            z_c_t = pyro.sample(f'z_c_{t}',
                                                pyro_dist.Categorical(theta_c_t),
                                                infer={"enumerate": "parallel"})
                            for attr_name in self._data_dims['DiscreteAttribute'].keys():
                                p_z_c_t = Vindex(phi[f'{attr_name}'])[z_c_t]
                                if attr_name == 'talkId':
                                    for j in range(self._data_dims['talkId_num']):
                                        obs_data = self._obs_data['Obs_attrs_clicked'][attr_name][j, :, t, batch]
                                        d_c_t_a = pyro.sample(f'd_c_{t}_{attr_name}_{j}',
                                                              pyro_dist.Categorical(p_z_c_t),
                                                              obs=obs_data)  # obs
                                else:
                                    obs_data = self._obs_data['Obs_attrs_clicked'][attr_name][:, t, batch]
                                    d_c_t_a = pyro.sample(f'd_c_{t}_{attr_name}',
                                                          pyro_dist.Categorical(p_z_c_t),
                                                          obs=obs_data)  # obs
                            obs_duration = self._obs_data['Obs_duration'][:, t, batch]
                            p_xi_1 = Vindex(xi_1)[z_c_t]
                            p_xi_2 = Vindex(xi_2)[z_c_t]
                            y_c_t = pyro.sample(f'y_c_{t}', pyro_dist.Gamma(p_xi_1, p_xi_2), obs=obs_duration)  # obs

    def run_svi(self):
        pyro.clear_param_store()
        guide = AutoDelta(
            poutine.block(self._model,
                          hide_fn=lambda msg: msg["name"].startswith("z_") or msg["name"].startswith("state_")))
        gamma = self._args.clipped_adam_gamma  # final learning rate will be gamma * self._args.learning_rate
        lrd = gamma**(1 / self._args.svi_step)
        optim = ClippedAdam({
            "lr": self._args.learning_rate,
            "betas": (self._args.betas_lower_bound, 0.999),
            "lrd": lrd
        })
        if self._args.num_particle:
            elbo = TraceEnum_ELBO(num_particles=self._args.num_particle, max_plate_nesting=self._max_plate_nesting)
        else:
            elbo = TraceEnum_ELBO(max_plate_nesting=self._max_plate_nesting)
        svi = SVI(self._model, guide, optim, elbo)
        for step in range(self._args.svi_step):
            loss = svi.step()
            logging.info(f"[{step + 1}/{self._args.svi_step}] loss = {loss}")


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser(
        description="Inference Algorithm Options")
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-2, type=float)
    parser.add_argument("--betas-lower-bound", default=0.95, type=float)
    parser.add_argument("--clipped-adam-gamma", default=0.1, type=float)
    parser.add_argument("-step", "--svi-step", default=1500, type=int)
    parser.add_argument("--num-particle", default=None, type=int)

    # model parameters
    userNum = 1000
    hyperParams = {"M": 10, "S": 3}
    dataDims = {
        "DiscreteAttribute": {
            "songId": 5000,
            "artistId": 3500,
            "contentId": 1500,
            "talkId": 144
        },
        "A": 5,
        "talkId_num": 3,
        "DemographicInfo": 6,
        "User": userNum,
        "Session": torch.randint(3, 10, (userNum, ))
    }

    # noinspection DuplicatedCode
    maxSessionNum = dataDims["Session"].max()

    browsedCards = [torch.randint(3, 10, (dataDims['Session'][i], )) for i in range(dataDims['User'])]
    browsedCardsPadding = pad_sequence(browsedCards, padding_value=1, batch_first=False)
    maxBrowsedCardsNum = browsedCardsPadding.max()

    # noinspection DuplicatedCode
    clickedCards = [torch.randint(3, 10, (dataDims['Session'][i], )) for i in range(dataDims['User'])]
    clickedCardsPadding = pad_sequence(clickedCards, padding_value=1, batch_first=False)
    maxClickedCardsNum = clickedCardsPadding.max()

    obsAttrsBrowsed = {}
    obsAttrsClicked = {}
    for attrName, attrDim in dataDims['DiscreteAttribute'].items():
        if attrName == 'talkId':
            obsAttrsBrowsed[attrName] = torch.randint(
                0, attrDim, (dataDims['talkId_num'], maxBrowsedCardsNum, maxSessionNum, userNum))
            obsAttrsClicked[attrName] = torch.randint(
                0, attrDim, (dataDims['talkId_num'], maxClickedCardsNum, maxSessionNum, userNum))
        else:
            obsAttrsBrowsed[attrName] = torch.randint(0, attrDim, (maxBrowsedCardsNum, maxSessionNum, userNum))
            obsAttrsClicked[attrName] = torch.randint(0, attrDim, (maxClickedCardsNum, maxSessionNum, userNum))

    obsDataSize = {
        "DemographicInfo": torch.rand((dataDims['User'], dataDims['DemographicInfo'])),
        "Browsed": browsedCardsPadding,
        "Clicked": clickedCardsPadding
    }

    obsData = {
        "Obs_attrs_browsed": obsAttrsBrowsed,
        "Obs_attrs_clicked": obsAttrsClicked,
        "Obs_duration": pyro_dist.Gamma(10., 1.).sample(torch.Size((maxClickedCardsNum, maxSessionNum, userNum)))
    }

    model = MinimalExample(hyper_params=hyperParams,
                           data_dims=dataDims,
                           obs_data=obsData,
                           obs_data_size=obsDataSize,
                           arg_parser=parser)
    model.run_svi()
