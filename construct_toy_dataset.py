"""Thermal dataset generator: FiPy geometry, cached SciPy finite-volume solves. SI units.
   Author: Will Little 10/30/26
"""
from time import perf_counter

# Report slow library loading when this file is run directly.
_import_start = perf_counter()
if __name__ == "__main__":
    print("[ThermoFlow] Loading NumPy and FiPy...", flush=True)

import numpy as np
from fipy import Grid3D, CellVariable
from scipy.sparse import coo_matrix, eye
from scipy.sparse.linalg import splu

def _log(message, verbose=True):
    """Flush messages immediately, including in redirected output."""
    if verbose:
        print(f"[ThermoFlow] {message}", flush=True)


CELL_SIZE = 0.0005
FRAME_TIMES = np.linspace(0.0, 60.0, 20)  # seconds since heating ends


def createCell(nx=48, ny=48, nz=12, *, n_cubes=2, cube_cells=4,
               seed=None, k_background=0.20, k_cube=0.50, verbose=False):
    """Return (mesh, k) for a heterogeneous and isotropic slab.

    Cubes are non-overlapping. Two background-cell layers separate cubes from exterior
    faces. Default geometry: 24 x 24 x 6 mm, with two 2 mm cubes.
    """
    dims = (nx, ny, nz)

    _log(f"Creating {nx} x {ny} x {nz} grid ({nx * ny * nz:,} cells); seed={seed}.", verbose)
    _log(f"Slab: {nx * CELL_SIZE * 1000:g} x {ny * CELL_SIZE * 1000:g} x "
         f"{nz * CELL_SIZE * 1000:g} mm; cell width: {CELL_SIZE * 1000:g} mm.", verbose)

    # Use cubic cells and (z, y, x) storage, with x varying fastest.
    mesh = Grid3D(nx=nx, ny=ny, nz=nz,
                  dx=CELL_SIZE, dy=CELL_SIZE, dz=CELL_SIZE)
    mask = np.zeros((nz, ny, nx), dtype=bool)
    rng = np.random.default_rng(seed)
    skin, gap = 2, 1
    if n_cubes and min(dims) < cube_cells + 2 * skin:
        raise ValueError("The cube does not fit inside the background skin.")

    # Rejection-sample cube locations while preserving the skin and gaps.
    for cube_index in range(n_cubes):
        for attempt in range(10000):
            ix, iy, iz = [int(rng.integers(skin, n - skin - cube_cells + 1))
                          for n in dims]
            # Include a one-cell margin when checking for nearby cubes.
            neighbors = mask[
                iz-gap:iz+cube_cells+gap,
                iy-gap:iy+cube_cells+gap,
                ix-gap:ix+cube_cells+gap,
            ]
            # Accept only placements that neither overlap nor touch existing cubes.
            if not neighbors.any():
                mask[iz:iz+cube_cells, iy:iy+cube_cells,
                     ix:ix+cube_cells] = True
                _log(f"Cube {cube_index + 1}/{n_cubes}: lower corner "
                     f"({ix * CELL_SIZE * 1000:g}, {iy * CELL_SIZE * 1000:g}, "
                     f"{iz * CELL_SIZE * 1000:g}) mm; "
                     f"side={cube_cells * CELL_SIZE * 1000:g} mm; "
                     f"placement attempts={attempt + 1}.", verbose)
                break
        else:
            raise RuntimeError("Could not place all cubes. Reduce count or size.")

    # Assign a scalar conductivity to each cell: heterogeneous but locally isotropic.
    values = np.where(mask, k_cube, k_background)
    k = CellVariable(name="conductivity", mesh=mesh,
                     value=values.ravel(), rank=0)
    _log(f"Conductivity: background={k_background:g}, cubes={k_cube:g} W/(m K); "
         f"inclusion cells={int(mask.sum()):,}.", verbose)
    return mesh, k


def _heat_operator(conductivity, cell_size, heat_capacity):
    """Assemble a conservative positive diffusion operator (units 1/s).

    Interior faces exchange heat with harmonic conductivity. Exterior faces
    have zero flux; prescribed heating is added separately to the RHS.
    """
    ids = np.arange(conductivity.size).reshape(conductivity.shape)
    rows, cols, entries = [], [], []
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(None, -1), slice(1, None)
        low, high = tuple(low), tuple(high)
        i, j = ids[low].ravel(), ids[high].ravel()
        ki, kj = conductivity[low].ravel(), conductivity[high].ravel()
        # Face conductance divided by cell heat capacity, on a uniform grid.
        rate = (2.0 * ki * kj / (ki + kj)) / (heat_capacity * cell_size**2)
        rows.extend((i, j, i, j))
        cols.extend((i, j, j, i))
        entries.extend((rate, rate, -rate, -rate))
    return coo_matrix((np.concatenate(entries),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(conductivity.size, conductivity.size)).tocsc()


def simulateVideo(nx=48, ny=48, nz=12, dt=0.5, *, n_cubes=2,
                  cube_cells=4, seed=None, heat_flux=800.0,
                  heat_capacity=1.5e6, verbose=True, log_every=20):
    """Heat z=0 for 5 seconds, then record 20 post-heating frames.

    Set verbose=False to suppress progress messages. log_every controls how
    frequently completed internal solves are reported (not camera frames).

    Returns
    video : ndarray, shape (20, ny, nx)
        Surface temperature rise in kelvin, relative to the initial temperature.
        Frames occur at FRAME_TIMES, including tau=0 immediately after heat switch-off
    heatField : ndarray, shape (nz, ny, nx)
        Conductivity in W/(m K), NOT a temperature field.

    All faces are insulated after the pulse. Surface cooling is redistribution
    into the slab; convection and radiation are not included. Backward Euler
    uses internal steps <= dt and lands exactly on pulse/frame boundaries.
    The conservative sparse operator is assembled once per specimen. LU factors
    are cached per distinct step size; the output API matches the FiPy reference.
    """
    if not np.isfinite([dt, heat_capacity, heat_flux]).all():
        raise ValueError("Simulation parameters must be finite.")
    if dt <= 0 or heat_capacity <= 0 or heat_flux < 0:
        raise ValueError("Require dt > 0, heat_capacity > 0, heat_flux >= 0.")
    if not isinstance(log_every, (int, np.integer)) or log_every < 1:
        raise ValueError("log_every must be a positive integer.")
    started = perf_counter()
    _log("Starting simulation.", verbose)
    mesh, k = createCell(nx, ny, nz, n_cubes=n_cubes,
                         cube_cells=cube_cells, seed=seed, verbose=verbose)
    # Retain FiPy geometry/material construction, but avoid rebuilding its equation
    # and refactorizing the same matrix during every time step.
    conductivity = np.asarray(k.value).reshape(nz, ny, nx)
    _log("Assembling the conservative sparse heat operator...", verbose)
    operator = _heat_operator(conductivity, CELL_SIZE, heat_capacity)
    identity = eye(conductivity.size, format="csc")
    temperature = np.zeros(conductivity.size, dtype=np.float64)

    # The front-cell source q/(C*dx) injects exactly q*surface_area watts.
    source = np.zeros((nz, ny, nx), dtype=np.float64)
    source[0] = heat_flux / (heat_capacity * CELL_SIZE)
    source = source.ravel()
    factors = {}
    cache_hits = 0
    time = 0.0
    completed_steps = 0
    _log(f"Solver ready: cached sparse LU, backward Euler, maximum dt={dt:g} s; "
         f"volumetric heat capacity={heat_capacity:g} J/(m³ K).", verbose)

    def advance(target):
        nonlocal time, completed_steps, temperature, cache_hits
        while target - time > 1e-10:
            step = min(dt, target - time)
            # Merge floating-point roundoff in nominally identical frame-end steps.
            cached_step = next((h for h in factors
                                if np.isclose(h, step, rtol=1e-11, atol=0.0)), None)
            step_started = perf_counter()
            if cached_step is None:
                cached_step = step
                _log(f"Factorizing matrix for dt={step:.9g} s "
                     f"({len(factors) + 1} cached step sizes)...", verbose)
                factors[cached_step] = splu(identity + cached_step * operator)
                _log(f"Factorization took {perf_counter() - step_started:.2f} s.", verbose)
            else:
                cache_hits += 1
            # Only the RHS changes while conductivity and step size remain fixed.
            rhs = temperature + cached_step * source
            temperature = factors[cached_step].solve(rhs)
            time += step
            completed_steps += 1
            if completed_steps == 1 or completed_steps % log_every == 0:
                _log(f"Solve {completed_steps}: physical t={time:.3f} s; "
                     f"step took {perf_counter() - step_started:.3f} s; "
                     f"elapsed={perf_counter() - started:.1f} s.", verbose)
        time = float(target)

    # Simulate heating before recording: the interior affects this initial response.
    _log(f"Heating z=0 at {heat_flux:g} W/m² for 5 s.", verbose)
    advance(5.0)
    expected_energy = heat_flux * (nx * CELL_SIZE) * (ny * CELL_SIZE) * 5.0
    measured_energy = heat_capacity * np.dot(temperature, mesh.cellVolumes)
    _log(f"Heating energy: expected={expected_energy:.6g} J; "
         f"measured={measured_energy:.6g} J.", verbose)
    if not np.isclose(measured_energy, expected_energy, rtol=1e-6, atol=1e-10):
        raise RuntimeError("Heating energy check failed.")

    # Switch off heating; all faces are now insulated.
    source.fill(0.0)
    _log(f"Heater off. Recording {len(FRAME_TIMES)} frames over "
         f"{FRAME_TIMES[-1]:g} s of redistribution.", verbose)
    video = []
    for frame_index, tau in enumerate(FRAME_TIMES):
        frame_started = perf_counter()
        _log(f"Frame {frame_index + 1}/{len(FRAME_TIMES)}: "
             f"advancing to τ={tau:.3f} s after switch-off.", verbose)
        advance(5.0 + float(tau))
        volume = temperature.reshape(nz, ny, nx)
        # Quadratic boundary extrapolation from cell averages with zero flux.
        # The first two z layers are homogeneous background by construction.
        surface = (7.0 * volume[0] - volume[1]) / 6.0
        # Copy the frame so later solver updates cannot modify saved observations.
        video.append(surface.copy())
        _log(f"Frame {frame_index + 1} saved: surface rise "
             f"min={surface.min():.5f}, mean={surface.mean():.5f}, "
             f"max={surface.max():.5f} K; frame took "
             f"{perf_counter() - frame_started:.2f} s.", verbose)

    final_energy = heat_capacity * np.dot(temperature, mesh.cellVolumes)
    # Insulated redistribution must preserve the energy injected during heating.
    _log(f"Final energy={final_energy:.6g} J; "
         f"energy error={final_energy - expected_energy:+.3e} J.", verbose)
    if not np.isclose(final_energy, expected_energy, rtol=1e-6, atol=1e-10):
        raise RuntimeError("Post-heating energy conservation check failed.")
    heatField = np.asarray(k.value).reshape(nz, ny, nx).copy()
    # Store compact arrays; the numerical solves above use double precision.
    video = np.asarray(video, dtype=np.float32)
    heatField = heatField.astype(np.float32)
    _log(f"Finished in {perf_counter() - started:.1f} s using {completed_steps} solves. "
         f"Factorizations={len(factors)}; cache hits={cache_hits}. "
         f"Video shape={video.shape}; conductivity shape={heatField.shape}.", verbose)
    return video, heatField


def _generate_one(job):
    """Run in the parent or a spawned worker; write before returning small metadata."""
    import os
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    index, filename, config, params, verbose = job
    path = Path(filename)
    started = perf_counter()
    video, field = simulateVideo(**params, verbose=verbose)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".sample-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, video=video, conductivity=field,
                                frame_times=FRAME_TIMES, cell_size=CELL_SIZE,
                                seed=params["seed"], sample_index=index,
                                n_cubes=params["n_cubes"], config_json=config)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path.name, perf_counter()-started


def generate_dataset(count=1, output="dataset", *, seed=0, start_index=0,
                     nx=48, ny=48, nz=12, dt=.5, n_cubes=2, cube_cells=4,
                     heat_flux=800., heat_capacity=1.5e6, resume=False,
                     verbose=False, workers=1, cube_count_range=None):
    """Generate independent samples, with cached solves and optional worker processes.

    Sample index i uses seed + i, so resuming or extending a batch preserves
    specimen identity. Existing files are never overwritten. --resume verifies
    saved settings and arrays before skipping a completed sample.
    cube_count_range=(low, high) selects an inclusive uniform cube count per
    specimen using a separate deterministic random stream; it overrides n_cubes.
    """
    import json
    import os
    from pathlib import Path

    for name, value, minimum in (("count", count, 1), ("workers", workers, 1), ("start_index", start_index, 0),
                                 ("seed", seed, 0), ("nx", nx, 1), ("ny", ny, 1),
                                 ("nz", nz, 2), ("n_cubes", n_cubes, 0),
                                 ("cube_cells", cube_cells, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if cube_count_range is not None:
        if (len(cube_count_range) != 2 or any(
                not isinstance(v, (int, np.integer)) or v < 0 for v in cube_count_range)
                or cube_count_range[0] > cube_count_range[1]):
            raise ValueError("cube_count_range must be two ordered nonnegative integers")
        cube_count_range = tuple(int(v) for v in cube_count_range)
    maximum_cubes = n_cubes if cube_count_range is None else cube_count_range[1]
    # Reject impossible cell-volume requests early. This is a necessary bound,
    # not a packing proof; createCell enforces the actual skin and gap constraints
    # and reports placement failures rather than silently reducing the count.
    interior = np.array([nx, ny, nz])-4
    if maximum_cubes and (np.any(interior < cube_cells)
                          or maximum_cubes*cube_cells**3 > np.prod(np.maximum(interior, 0))):
        raise ValueError("Requested cubes cannot fit inside the background skin")
    if not np.isfinite([dt, heat_flux, heat_capacity]).all() or dt <= 0 or heat_flux < 0 or heat_capacity <= 0:
        raise ValueError("Require finite dt > 0, heat_flux >= 0, heat_capacity > 0")
    if maximum_cubes and min(nx, ny, nz) < cube_cells + 4:
        raise ValueError("Cubes do not fit inside the two-cell background skin")
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(i, output / f"sample_{i:04d}.npz")
            for i in range(start_index, start_index + count)]
    if not resume:
        for _, path in jobs:
            if path.exists():
                raise FileExistsError(f"{path} already exists; use --resume or a new output folder")
    settings = dict(schema_version=1, nx=nx, ny=ny, nz=nz, dt=dt,
                    n_cubes=n_cubes, cube_cells=cube_cells, heat_flux=heat_flux,
                    heat_capacity=heat_capacity, pulse_seconds=5.,
                    k_background=.20, k_cube=.50, cell_size=CELL_SIZE,
                    frame_times=FRAME_TIMES.tolist(),
                    solver="Cached sparse LU backward Euler")
    if cube_count_range is not None:
        settings.pop("n_cubes")
        settings.update(cube_count_range=list(cube_count_range),
                        cube_count_sampling="uniform_seedsequence_v1")
    # Validate all existing samples before scheduling new work.
    jobs_to_run = []
    skipped = 0
    for index, path in jobs:
        sample_seed = seed+index
        # Keep count sampling independent from the cube-position RNG. Results
        # are unchanged by worker count, task order, or resuming a partial run.
        selected_cubes = n_cubes if cube_count_range is None else int(
            np.random.default_rng(np.random.SeedSequence([sample_seed, 0x54464C57])).integers(
                cube_count_range[0], cube_count_range[1]+1))
        specimen_settings = dict(settings, n_cubes=selected_cubes, seed=sample_seed)
        params = dict(nx=nx, ny=ny, nz=nz, dt=dt, n_cubes=selected_cubes,
                      cube_cells=cube_cells, seed=seed+index,
                      heat_flux=heat_flux, heat_capacity=heat_capacity)
        config = json.dumps(specimen_settings, sort_keys=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                saved = json.loads(str(data["config_json"].item())) if "config_json" in data else {}
                # These implementations use the same discrete equations. Accept the
                # validated FiPy predecessor so existing datasets can be extended.
                known_solvers = {"FiPy backward Euler LinearLUSolver", "Cached sparse LU backward Euler"}
                equivalent = saved.get("solver") in known_solvers
                saved["solver"] = settings["solver"]
                if (not equivalent or saved != specimen_settings
                        or data["video"].shape != (len(FRAME_TIMES), ny, nx)
                        or data["conductivity"].shape != (nz, ny, nx)
                        or not np.array_equal(data["frame_times"], FRAME_TIMES)
                        or float(data["cell_size"]) != CELL_SIZE or int(data["seed"]) != seed+index
                        or not np.isfinite(data["video"]).all()
                        or not np.isfinite(data["conductivity"]).all()):
                    raise ValueError(f"{path}: incompatible or incomplete sample; use a new folder")
            skipped += 1
            _log(f"Skip {path.name}: verified existing sample ({skipped} skipped).")
        else:
            jobs_to_run.append((index, str(path), config, params, verbose))
    started = perf_counter()
    completed = 0
    _log(f"Batch: {count} requested, {skipped} existing, {len(jobs_to_run)} to generate; "
         f"workers={workers}; output={output.resolve()}")

    def report(result):
        nonlocal completed
        name, duration = result
        completed += 1
        elapsed = perf_counter()-started
        remaining = len(jobs_to_run)-completed
        _log(f"Saved {name}: specimen took {duration:.1f} s; "
             f"progress {completed+skipped}/{count}; elapsed {elapsed/60:.1f} min; "
             f"estimated remaining {elapsed/completed*remaining/60:.1f} min.")

    if workers == 1:
        for job in jobs_to_run:
            report(_generate_one(job))
    elif jobs_to_run:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        # Set before spawning: numerical libraries initialize as workers import
        # this module. One library thread per worker avoids CPU oversubscription.
        thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                       "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in thread_vars}
        try:
            for name in thread_vars:
                os.environ[name] = "1"
            with ProcessPoolExecutor(max_workers=min(workers, len(jobs_to_run)),
                                     mp_context=get_context("spawn")) as pool:
                futures = [pool.submit(_generate_one, job) for job in jobs_to_run]
                try:
                    for future in as_completed(futures):
                        report(future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    _log(f"Batch complete: {completed} generated, {skipped} skipped; "
         f"elapsed {(perf_counter()-started)/60:.1f} min.")
    return [path for _, path in jobs]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate thermal videos and conductivity fields in batches.")
    parser.add_argument("--count", type=int, default=1, help="Number of samples (default: 1)")
    parser.add_argument("--output", default="dataset", help="Destination folder (default: dataset)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; specimen i uses seed+i")
    parser.add_argument("--start-index", type=int, default=0, help="First sample index")
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument("--cubes", dest="n_cubes", type=int, default=2,
                        help="Fixed cubes per specimen (default: 2)")
    counts.add_argument("--cube-count-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="Uniform inclusive cube-count range, sampled independently for each specimen")
    parser.add_argument("--cube-cells", type=int, default=4, help="Cube side length in cells")
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--dt", type=float, default=.5, help="Maximum internal timestep, seconds")
    parser.add_argument("--heat-flux", type=float, default=800., help="Heating flux, W/m²")
    parser.add_argument("--heat-capacity", type=float, default=1.5e6, help="Volumetric heat capacity, J/(m³ K)")
    parser.add_argument("--workers", type=int, default=1, help="Independent specimen processes (try 2, then 4; default: 1)")
    parser.add_argument("--resume", action="store_true", help="Verify and skip matching existing samples")
    parser.add_argument("--verbose", action="store_true", help="Also log each specimen's simulation progress")
    args = parser.parse_args()
    _log(f"Libraries loaded in {perf_counter()-_import_start:.1f} s.")
    generate_dataset(**vars(args))


if __name__ == "__main__":
    main()
