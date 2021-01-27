from logging import getLogger
from pathlib import Path
import random
import numpy as np
import h5py as h5
import yaml
import isle

# This script is a fork of Jan-Lukas Wynens make-train-data.py in foldilearn.
# foldilearn introduced Machine Learning parametrization of Lefschetz thimbles
# using real valued neuronal networks in the first iteration. I simply take up
# this work and develop it using complex valued neuronal networks allowing for
# a greater scalability.

# ==============================================================================
# Parameters
# ==============================================================================
# name the path where you want to store the data. Should exist!
DIR = Path(__file__).resolve().parent
DATADIR = DIR/"./data/"

# define the lattice you want to use. Note the isle documentation
# https://evanberkowitz.github.io/isle/
LATTICE = "triangle"

# define the model parameters for the (hubbard) model in isle
PARAMS = isle.util.parameters(
    beta=6,
    U=4,
    mu=0,
    sigmaKappa=-1,
    hopping=isle.action.HFAHopping.EXP
)

# define the temporal direction discretization
Nt = 16

# define the number of training data points (data,label) you want to create
NSAMPLES = 10_000

# define the parameters for the flow
FLOW_PARAMS = isle.util.parameters(
    flowTime=0.05*32/Nt,
    minFlowTime=0.01*32/Nt,
    stepSize=5e-4,
    starty=None
)

# ==============================================================================
# Program definitions, don't change anything below here unless you know what you
# are doing
# ==============================================================================

def randomSigns(rng, n):
    return rng.uniform(0, 2, n).astype(int)*2-1

def makeStartPhi(lat, params, starty, rng):
    """
    Generate a single configuration on the real plane.
    """

    sigma = rng.uniform(np.sqrt(params.tilde("U", lat)/(1+16/lat.nt())), np.sqrt(params.tilde("U", lat)/1.0), 1)[0]

    return isle.Vector(np.random.normal(0, sigma, lat.lattSize())
                       + 0j), sigma

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

def initFile(lattice, params, flowParams, overwrite):
    # Pattern to format file names using str.format.
    OUTFILE_FMT = "{kind}.{latname}.{hopping}.nt{nt}.beta{beta}.U{U}.mu{mu}{extra}.h5"

    def formatFloat(x):
        """!
        Format a single float for file names.
        """
        return f"{float(x)}".replace(".", "_")

    def formatFname(kind, lattice, params, extra=None, nt=None):
        """!
        Generate the name including path fo an output file.
        """

        if extra is not None and not isinstance(extra, str):
            if hasattr(extra, "flowTime"):
                extra = "flow"+formatFloat(extra.flowTime)
            else:
                extra = str(extra)
        if not isinstance(lattice, str):
            nt = lattice.nt()

        return OUTFILE_FMT.format(
            kind=kind,
            latname=lattice.name.replace(" ", "_") if isinstance(lattice, isle.Lattice) else lattice,
            hopping="exp" if params.hopping == isle.action.HFAHopping.EXP else "dia",
            nt=nt,
            beta=formatFloat(params.beta),
            U=formatFloat(params.U),
    		mu=formatFloat(params.mu),
            extra=f".{extra}" if extra else "")

    fname = DATADIR/formatFname(f"{NSAMPLES}_train", lattice, params, flowParams)
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

def flowStepwise(phi, lattice, action, tflow, rkStepSize, nsteps=10):
    tflow = tflow / nsteps

    for i in range(nsteps):
        phi, actVal, actualFlowTime = isle.rungeKutta4Flow(phi, action,
                                                           tflow, rkStepSize,
                                                           imActTolerance=1e-6)
        if actualFlowTime < tflow:
            # didn't reach final point in flow, can't go on
            if actualFlowTime == 0.0 or i == 0:
                raise RuntimeError("Failed to flow")
            break

    return phi, actVal

def preprocessFlowParams(lattice, params):
    if FLOW_PARAMS.starty is None:
        import foldilearn.resources as flr
        return FLOW_PARAMS.replace(starty=flr.loadCriticalPoint(lattice, params.beta, params.U, 0)[0][0].imag)
    return FLOW_PARAMS

def generate(overwrite):
    """
    Generate and store a single dataset.
    """

    log = getLogger(f"{__name__}")

    lattice = isle.LATTICES[LATTICE]
    lattice.nt(Nt)
    params = PARAMS
    flowParams = preprocessFlowParams(lattice, params)

    outfname = initFile(lattice, params, flowParams, overwrite)
    # The default RNG should not have been seeded...
    rng = isle.random.NumpyRNG(random.randint(0, 10000))
    action = makeAction(lattice, params)

    flowTime = []
    width = []

    conf_gauss = np.empty((NSAMPLES, lattice.lattSize()), dtype=complex)
    conf_flowed = np.empty((NSAMPLES, lattice.lattSize()), dtype=complex)
    actions = np.empty(NSAMPLES, dtype=complex)
    with isle.cli.trackProgress(NSAMPLES) as pbar:
        isample = 0
        tries = 0
        while isample < NSAMPLES:
            tries += 1
            phi_gauss, sigma = makeStartPhi(lattice, params, flowParams.starty, rng)
            phi_flowed, actVal, actualFlowTime = isle.rungeKutta4Flow(phi_gauss, action,
                                                               flowParams.flowTime,
                                                               flowParams.stepSize,
                                                               imActTolerance=1e-6)
            if actualFlowTime < flowParams.minFlowTime:
                log.info("Failed to generate phi at sample %d, reached flow time: %f",
                         isample, actualFlowTime)
            else:
                conf_gauss[isample, :] = phi_gauss
                conf_flowed[isample, :] = phi_flowed
                actions[isample] = actVal
                flowTime.append(actualFlowTime)
                width.append(sigma)
                isample += 1
                pbar.advance()

            pbar._message = f"Success rate: {isample/tries:.3f}"

    with h5.File(str(outfname), "a") as h5f:
        h5f["phi_flowed"] = conf_flowed
        h5f["phi_gauss"] = conf_gauss
        h5f["action_flowed"] = actions
        grp = h5f.create_group("flow_diagnostics")
        grp["flow_time"] = np.array(flowTime)
        grp["width"] = np.array(width)


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