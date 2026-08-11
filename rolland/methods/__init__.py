# ------------------------------------------------------------------------------
# Rolland
# ------------------------------------------------------------------------------

"""The Rolland library: several classes for the implementation of rolling noise calculation."""


from .analytical import (
    AnalyticalMethods,
    EBBCont1LSupp,
    EBBCont2LSupp,
    TSDiscr1LSupp,
    TSDiscr2LSupp,
)

__all__ = ["AnalyticalMethods",
           "EBBCont1LSupp",
           "TSDiscr2LSupp",
           "TSDiscr1LSupp",
           "EBBCont2LSupp",
           ]




