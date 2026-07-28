if __name__ == "__main__":
    from optisample.threads import bind_compute_threads

    bind_compute_threads()

    from optisample.cli import main

    main()
