# ------------------------------------------------------------------------------
# Rolland
# ------------------------------------------------------------------------------

"""The Rolland library: several classes for the implementation of rolling noise calculation."""


from .analytical import (
    EBBCont1L,
    EBBCont2L,
    TSDiscr1LSupp,
    TSDiscr2LSupp,
)

__all__ = ["EBBCont1L",
           "TSDiscr2LSupp",
           "TSDiscr1LSupp",
           "EBBCont2L",
           ]




