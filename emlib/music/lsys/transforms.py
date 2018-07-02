from .core import *


def transform_duration(mode='dur') -> NodeTransform:
    """
    This transform uses the following directives:

    %x  : modify all durations by multiplying them by 'x'
    %/x : modify all durations by dividing them by 'x'

    mode:
        'dur'    -> the modifications are applied to the durations of the node
        'weight' -> the modifications are applied to the weight of the node

    It can be used as:

      A) transform_step().apply(branch)
      B) branch.transform(*transform_step())
      C) LSystem(..., transforms=[transform_step()])

    """
    def callback(node:Node, state:Dict[str, Any]):
        if node.name[0] == "%":
            tok, factor = parse_token(node.name, 1)
            if tok == "%/":
                factor = 1/factor
            state['durfactor'] *= factor
        elif node.weight > 0:
            if mode == 'dur':
                return node.clone(dur=node.weight*state['durfactor'])
            elif mode == 'weight':
                return node.clone(dur=node.dur*state['durfactor'])
            else:
                raise ValueError(mode)
    return NodeTransform(callback, state={'durfactor': 1})


def transform_step(startstep:float=60, heightattr='h') -> NodeTransform:
    """
    Directives: this transform uses the following directives:

    +x  : add 'x' to current step                     (step += x * stepfactor)
    -x  : substract 'x' to current step               (step -= x * stepfactor)
    *x  : modify all steps by multiplying them by 'x' (stepfactor *= x)
    /x  : modify all steps by dividing them by 'x'    (stepfactor /= x)

    startstep: the step to start with (can also be set at the axiom as @(step=60)

    heightattr: used to define the height of a node
        For example, the rule 'A': 'B(h=1)' defines a node B with height 1
        as substitution for A.

    startstep: the step to start with

    NB: global state directives can be inserted like @(step=60), for instance
        as part of the axiom of the LSystem

    It can be used as:

      A) transform_step().apply(branch)
      B) branch.transform(*transform_step())
      C) LSystem(..., transforms=[transform_step()])
    """
    def callback(node:Node, state:Dict[str, Any]):
        if node.weight == 0 and node.name[0] in "*/+-":
            tok, value = parse_token(node.name, 1)
            if tok == '*':
                state['stepfactor'] *= value
            elif tok == "/":
                state['stepfactor'] /= value
            elif tok == '+':
                state['step'] += value * state['stepfactor']
            elif tok == '-':
                state['step'] -= value * state['stepfactor']
        else:
            step = state['step']
            height = node.data.get(heightattr)
            if height is None:
                return node.clone(step=step)
            height *= state['stepfactor']
            state["step"] = step + height
            return node.clone(step=step, stepend=step+height, data={heightattr: height})
    state = {'stepfactor': 1}
    if startstep is not None:
        state['step'] = startstep
    return NodeTransform(callback, state=state)

