import argparse


def add_bool_arg(parser, name, default=False, **kwargs):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--" + name,
        dest=name,
        action="store_true",
        help="Default: " + ("Enabled" if default else "Disabled"),
    )
    group.add_argument("--no-" + name, dest=name, action="store_false", **kwargs)
    parser.set_defaults(**{name: default})
