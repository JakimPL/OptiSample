from optisample.threads import bind_compute_threads

# pytest loads the plugins named in ``addopts`` before it imports any conftest or test module, which is the
# last moment the numerical libraries can be held to one thread each: each reads its count once, as it is
# imported. Every xdist worker is a process of its own and reads the same ``addopts``, so each is held too.
bind_compute_threads()
