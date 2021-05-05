"""
Generate training data sets for supervised learning using holomorphic flow.
"""


from logging import getLogger
from pathlib import Path
import random

import numpy as np
import h5py as h5
import yaml

import isle
import isle.drivers

from foldilearn.fileio import formatFname

DIR = Path(__file__).resolve().parent
DATADIR = DIR/"data"

LATTICE = "four_sites"

PARAMS = isle.util.parameters(
    beta=5,
    U=2,
    mu=4,
    sigmaKappa=-1,
    hopping=isle.action.HFAHopping.EXP)
NT = 16

NSAMPLES = 10000

FLOW_PARAMS = isle.util.parameters(
    flowTime=0.05*32/NT,
    minFlowTime=0.01*32/NT,
    stepSize=5e-4,
    starty=None
)

def makeAction(lattice, params):
    """
    Make an action.
    """
    # Import everything this function needs so it is self-contained.
    import isle.action

    return isle.action.HubbardGaugeAction(params.tilde("U", lattice)) \
        + isle.action.makeHubbardFermiAction(lattice,
                                             params.beta,
                                             params.tilde("mu", lattice),
                                             params.sigmaKappa,
                                             params.hopping)

def preprocessFlowParams(lattice, params):
    if FLOW_PARAMS.starty is None:
        import foldilearn.resources as flr
    #    return FLOW_PARAMS.replace(starty=flr.loadCriticalPoint(lattice, params.beta, params.U, 0)[0][0].imag)
    return FLOW_PARAMS

def initFile(lattice, params, flowParams, overwrite):
    fname = DATADIR/formatFname("train", lattice, params, flowParams)
    log = getLogger(__name__)
    if fname.exists():
        if overwrite:
            log.info("Output file %s exists -- overwriting", fname)
            fname.unlink()
        else:
            log.error("Output file %s exists and not allowed to overwrite", fname)
            raise RuntimeError("Ouput file exists")

    isle.h5io.writeMetadata(fname, lattice, params, isle.meta.sourceOfFunction(makeAction))
    with h5.File(str(fname), "a") as h5f:
        h5f["meta/flowParams"] = yaml.dump(flowParams)

    return fname

def generate(overwrite):
    """
    Generate and store a single dataset.
    """

    log = getLogger(f"{__name__}")

    lattice = isle.LATTICES[LATTICE]
    lattice.nt(NT)
    params = PARAMS
    flowParams = preprocessFlowParams(lattice, params)

    outfname =  initFile(lattice,params,flowParams,overwrite) #DATADIR/formatFname("train", lattice, params, flowParams)

    # The default RNG should not have been seeded...
    rng = isle.random.NumpyRNG(random.randint(0, 10000))

    action = makeAction(lattice, params)

    # Set up a fresh HMC driver.
    # It handles all HMC evolution as well as I/O.
    # Last argument forbids the driver to overwrite any existing data.
    #hmcState = isle.drivers.hmc.newRun(lat, params, rng, makeAction, outfile_fn, overwrite)
    hmcStage = isle.drivers.hmc.HMC(lattice,params,rng,action,outfname,0)

    # Generate a random initial condition.
    # Note that configurations must be vectors of complex numbers.
    phi_unflowed = isle.Vector(rng.normal(0, params.tilde("U", lattice)**(1/2), lattice.lattSize())+0j)

    log.info("Thermalizing")
    # Pick an evolver which linearly decreases the number of MD steps from 20 to 5.
    # The number of steps (99) must be one less than the number of trajectories below.
    evolver = isle.evolver.LinearStepLeapfrog(action, (1, 1), (20, 5), 99, rng)
    phi_unflowed = hmcStage(phi_unflowed,evolver,1000,0,0).phi

    # Run production.
    log.info("Producing")
    # Pick a new evolver with a constant number of steps to get a reproducible ensemble.
    evolver = isle.evolver.ConstStepLeapfrog(action, 1, 5, rng)
    # Produce configurations and save in intervals of 2 trajectories.
    # Place a checkpoint every 10 trajectories.

    flowTime = []

    fields_flowed = np.empty((NSAMPLES, lattice.lattSize()), dtype=complex)
    fields_unflowed = np.empty((NSAMPLES, lattice.lattSize()), dtype=complex)
    actions = np.empty(NSAMPLES, dtype=complex)

    isample = 0
    tries = 0
    while isample < NSAMPLES:
        tries += 1
        # create the next 100 trajectories and return the last as new configuration
        phi_unflowed = hmcStage(phi_unflowed,evolver,100,0,0).phi

        # Solve the flow equation
        phi_flowed, actVal, actualFlowTime = isle.rungeKutta4Flow(phi_unflowed, action,
                                                               flowParams.flowTime,
                                                               flowParams.stepSize,
                                                               imActTolerance=1e-6)
        if actualFlowTime < flowParams.minFlowTime:
            log.info("Failed to generate phi at sample %d, reached flow time: %f",isample, actualFlowTime)
        else:
            # store the fields
            fields_flowed[isample, :] = phi_flowed
            fields_unflowed[isample, :] = phi_unflowed
            actions[isample] = actVal
            flowTime.append(actualFlowTime)
            isample += 1

    # write the data to file
    with h5.File(str(outfname), "a") as h5f:
        h5f["phi_flowed"] = fields_flowed
        h5f["phi_unflowed"] = fields_unflowed
        h5f["action_flowed"] = actions
        grp = h5f.create_group("flow_diagnostics")
        grp["flow_time"] = np.array(flowTime)

def main():
    parser = isle.cli.makeDefaultParser(description="Generate training datasets",
                                        defaultLog="none")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files.")
    clArgs = isle.initialize(parser)

    if not DATADIR.exists():
        DATADIR.mkdir()

    generate(clArgs.overwrite)


if __name__ == "__main__":
    main()
