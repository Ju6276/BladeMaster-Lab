"""Mapped external contact in the compensated interface momentum equation."""

import numpy as np

from ._cohesive_expansion import E, Equations, assign, divide


class ContactEquations(Equations):
    def __init__(self, *args, contacts, evaluate_contact, mass, duration, constrain_normal=True):
        super().__init__(*args, constrain_normal=constrain_normal)
        self.contacts = contacts
        self.evaluate_contact = evaluate_contact
        self.mass = E(mass)
        self.duration = E(duration)

    def contact_response(self, y):
        displacement = self.length * y[: self.size]
        results = []
        for contact in self.contacts:
            mapping = E.from_value(contact["map"])
            motion = mapping @ displacement
            local = self.evaluate_contact(
                contact,
                E.from_value(contact["point"]) + motion,
                E.from_value(contact["velocity"]) + divide(motion, self.duration),
                self.duration,
            )
            derivative = local.force_jacobian[:, :3] @ mapping
            derivative = derivative + divide(local.force_jacobian[:, 3:6] @ mapping, self.duration)
            results.append((mapping, local.force, derivative, local.history, local.mode, local.gap))
        return results

    def __call__(self, y):
        residual, jacobian, gaps = super().__call__(y)
        coefficient = divide(self.duration * self.duration, self.mass)
        for mapping, force, derivative, *_ in self.contact_response(y):
            assign(
                residual,
                slice(0, self.size),
                residual[: self.size] - divide(coefficient * (mapping.T @ force), self.length),
            )
            assign(
                jacobian,
                (slice(0, self.size), slice(0, self.size)),
                jacobian[: self.size, : self.size] - coefficient * (mapping.T @ derivative),
            )
        return residual, jacobian, gaps
