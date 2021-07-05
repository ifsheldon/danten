from typing import Optional

import optim
import taichi as ti


@ti.data_oriented
class Momentum(optim.Optimizer):
    def __init__(self,
                 lr: float,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0,
                 dampening: float = 0.0,
                 max_grad: Optional[float] = None):
        self.lr = ti.field(ti.f32, ())
        self.momentum = ti.field(ti.f32, ())
        self.weight_decay = ti.field(ti.f32, ())
        self.dampening = ti.field(ti.f32, ())
        self.max_grad = ti.field(ti.f32, ())
        self.lr[None] = lr
        self.momentum[None] = momentum
        self.weight_decay[None] = weight_decay
        self.dampening[None] = dampening
        self.fields = []
        if max_grad is not None:
            self.max_grad[None] = max_grad
        else:
            self.max_grad[None] = -1.0  # TODO: fix this

    def register_field(self, field):
        print(field.grad)
        self.fields.append(field)
        return self

    def step(self):
        pass


if __name__ == "__main__":
    ti.init(ti.cuda)
    field = ti.field(ti.f32, (), needs_grad=True)
    opt = Momentum(0.)
    opt.register_field(field)
