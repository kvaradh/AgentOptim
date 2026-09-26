"""Typed optimization outcomes that the product can render safely."""


class OptimizationHalt(RuntimeError):
    code = "optimization_halt"


class UnsupportedSeedError(ValueError):
    code = "unsupported_seed"


class UnsupportedTargetError(OptimizationHalt):
    code = "unsupported_target"


class ProposalExhaustedError(OptimizationHalt):
    code = "proposal_exhausted"


class NoSafeCandidateError(OptimizationHalt):
    code = "no_safe_candidate"
