#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   auto_guide_list_bug.py
@Time    :   2023/05/23 21:13:35
@Author  :   Jinnan Huang 
@Contact :   jinnan_huang@stu.xjtu.edu.cn
@Desc    :   None
"""
import sys

sys.path.append(".")

import pyro
import pyro.distributions as pyro_dist
import torch
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta, AutoGuideList, AutoNormal
from pyro.ops.indexing import Vindex
from pyro.optim import Adam


class AutoGuideListBug:
    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def model(self):
        m = 10
        m_dim = 2500
        motivation_plate = pyro.plate("motivations", m, dim=-1, device=self._device)
        eta_a = torch.ones((m_dim, ), device=self._device)
        with motivation_plate:
            phi = pyro.sample("phi", pyro_dist.Dirichlet(eta_a))

        session = torch.randint(1, 10, (1000, ), device=self._device)
        max_length = session.max()
        batch_size = 10
        with pyro.plate("batched_users", 1000, batch_size, dim=-1, device=self._device) as batch:
            session_length = session[batch]
            for t in pyro.markov(range(max_length)):
                with poutine.mask(mask=t < session_length):
                    theta_prior = F.softmax(torch.rand((batch_size, m), device=self._device), dim=-1)
                    theta = pyro.sample("theta_{}".format(t), pyro_dist.Dirichlet(theta_prior))
                    browsed_cards_t = torch.randint(1, 20, (batch_size, ), device=self._device)
                    max_browsed_cards_num_t = browsed_cards_t.max()
                    with pyro.plate(f"cards_{t}", max_browsed_cards_num_t, dim=-2,
                                    device=self._device) as cards_plate:
                        with poutine.mask(mask=cards_plate.unsqueeze(-1) < browsed_cards_t.unsqueeze(0)):
                            z = pyro.sample(f"z_{t}", pyro_dist.Categorical(theta), infer={"enumerate": "parallel"})
                            p_z = Vindex(phi)[z]
                            d_z = pyro.sample(f"d_{t}",
                                              pyro_dist.Categorical(p_z),
                                              obs=torch.randint_like(z, 0, m_dim))

    def trace_model(self):
        trace = poutine.trace(self.model).get_trace()
        trace.compute_log_prob()
        print(f"\n{trace.format_shapes()}")

    def run_svi(self, flag=None):
        guide = AutoGuideList(self.model)
        guide.append(AutoDelta(poutine.block(self.model, expose_fn=lambda msg: msg["name"].startswith("phi"))))
        guide.append(
            AutoNormal(
                poutine.block(self.model,
                              hide_fn=lambda msg: msg["name"].startswith("phi") or msg["name"].startswith("z"))))
        if flag == "AutoDelta":
            guide = AutoDelta(poutine.block(self.model, hide_fn=lambda msg: msg["name"].startswith("z")))
        optim = Adam({"lr": 0.001})
        elbo = TraceEnum_ELBO(max_plate_nesting=2)
        svi = SVI(self.model, guide, optim, elbo)
        for i in range(1000):
            loss = svi.step()
            print(f"loss in step {i+1}: {loss}")


if __name__ == "__main__":
    auto_guide_list_bug = AutoGuideListBug()
    auto_guide_list_bug.run_svi()  # if `flag` is `AutoDelta`, it will work properly.
