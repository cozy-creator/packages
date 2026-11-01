# Native attention-output ownership

The dense and sparse attention producers own the attended output until its one output projection completes, then release it. Declare that boundary explicitly instead of manufacturing `_cozy_siblings` learning. Generic calls retain their current sharing checks; masks, parallel routes and unsupported layouts remain outside this declaration. Actual first-use processor/Loop output and ownership checks are required before qualification. No Triton kernel source, entry identity or numerical recipe changes are planned.
