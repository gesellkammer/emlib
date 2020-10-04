import constraint

"""
class AllDifferentConstraint(Constraint):
    def __call__(self, variables, domains, assignments, forwardcheck=False,
                 _unassigned=Unassigned):
        seen = {}
        for variable in variables:
            value = assignments.get(variable, _unassigned)
            if value is not _unassigned:
                if value in seen:
                    return False
                seen[value] = True
        if forwardcheck:
            for variable in variables:
                if variable not in assignments:
                    domain = domains[variable]
                    for value in seen:
                        if value in domain:
                            domain.hideValue(value)
                            if not domain:
                                return False
        return True
"""

class NotAllEqualConstraint(constraint.Constraint):
    """
    Constraint enforcing that values of all given variables are NOT equal

    NOT WORKING
    """

    def __call__(self, variables, domains, assignments, forwardcheck=False,
                 _unassigned=constraint.Unassigned):
        singlevalue = _unassigned
        for variable in variables:
            value = assignments.get(variable, _unassigned)
            if singlevalue is _unassigned:
                singlevalue = value
            elif value is not _unassigned and value == singlevalue:
                return False
        if forwardcheck and singlevalue is not _unassigned:
            for variable in variables:
                if variable not in assignments:
                    domain = domains[variable]
                    if singlevalue in domain:
                        domain.hideValue(singlevalue)
        return True