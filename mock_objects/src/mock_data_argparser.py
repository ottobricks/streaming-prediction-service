import argparse
from argparse import ArgumentParser


def _create_toplevel_parser() -> ArgumentParser:
    """Create parent parser with shared arguments"""
    parser = ArgumentParser(
        description="Module to generate mock data for 'streaming-prediction-service' development"
    )
    parser.add_argument(
        "--prediction-target-type",
        choices=["binary", "multiclass", "regression"],
        required=True,
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--nrows", type=int, default=1_000)
    parser.add_argument("--nfeatures-poisson", type=int, default=5)
    parser.add_argument("--nfeatures-normal", type=int, default=5)
    parser.add_argument("--nfeatures-poisson-categorical", type=int, default=3)
    return parser


def _add_children_args(parser: ArgumentParser, child_t: str) -> ArgumentParser:
    """Add prediction-target specific arguments to parser"""
    child_args_map = {
        "binary": _add_binary_featureframe_generation_args,
        "multiclass": _add_multiclass_featureframe_generation_args,
        "regression": _add_regression_featureframe_generation_args,
    }
    return child_args_map[child_t](parser)


def _add_binary_featureframe_generation_args(parser: ArgumentParser) -> ArgumentParser:
    """Add binary-prediction-target specific arguments to parser"""

    def restricted_float(arg: float) -> float:
        try:
            arg = float(arg)
        except ValueError as excpt:
            raise argparse.ArgumentTypeError(f"Value '{arg}' is not a float.")
        if not (arg > 0.0 and arg < 1.0):
            raise argparse.ArgumentError(
                "--negative-class-proportion",
                "Value must be contained in interval (0., 1.)",
            )
        return arg

    parser.add_argument("--negative-class-proportion", type=restricted_float)
    return parser


def _add_multiclass_featureframe_generation_args(
    parser: ArgumentParser,
) -> ArgumentParser:
    raise NotImplementedError()


def _add_regression_featureframe_generation_args(
    parser: ArgumentParser,
) -> ArgumentParser:
    raise NotImplementedError()


def parse_args() -> dict:
    """Public interface to parse args for module MockData"""
    toplevel_parser = _create_toplevel_parser()
    toplevel_args = toplevel_parser.parse_args()

    parser = _add_children_args(toplevel_parser, toplevel_args.prediction_target_type)
    return vars(parser.parse_args())
