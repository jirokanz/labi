import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="labi", description="Labi")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Start the interactive autonomous agent (default)")
    sub.add_parser("demo", help="Run the small routing-decision demo")
    sub.add_parser("benchmark", help="Run the memory-router benchmark")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    command = args.command or "run"

    if command == "run":
        from labi.agent import main as agent_main
        agent_main()
    elif command == "demo":
        from labi.runtime import main as demo_main
        demo_main()
    elif command == "benchmark":
        from labi.benchmark.run import main as benchmark_main
        benchmark_main()


if __name__ == "__main__":
    main()
