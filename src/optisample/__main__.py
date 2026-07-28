from optisample.threads import bind_compute_threads


def main() -> None:
    """Run the command line with this process held to one compute thread.

    The limit is stated first and the CLI is reached from inside the call, since every numerical library
    behind it reads its thread count as it loads (:func:`~optisample.threads.bind_compute_threads`).
    """
    bind_compute_threads()

    from optisample.cli import main as run_cli  # pylint: disable=import-outside-toplevel

    run_cli()


if __name__ == "__main__":
    main()
