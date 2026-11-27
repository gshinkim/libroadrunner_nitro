/* Fit two rate constants with roadrunner_nitro's fitting C API (rrc_fit.h), from C. Exits 1 if the fit does
 * not recover the parameters the data were made from, so it doubles as an end-to-end test of the library.
 * Build and run from the repository folder:
 *
 *   cargo build --release -p roadrunner_nitro_c_api
 *   cc -I capi/include capi/examples/fit.c -L target/release -lroadrunner_nitro_c_api -lm \
 *      -Wl,-rpath,$PWD/target/release -o fit && ./fit
 *   (Windows, clang: clang -I capi/include capi/examples/fit.c target/release/roadrunner_nitro_c_api.dll.lib -o fit.exe,
 *    then run with target/release on PATH)
 *
 * The model: X0 -> S1 (k1 * X0), S1 -> (k2 * S1), X0 a boundary species. With S1(0) = 0,
 * S1(t) = k1 X0 / k2 (1 - exp(-k2 t)). Two conditions: X0 = 10 (the model's value) and X0 = 5.
 */
#include <math.h>
#include <stdio.h>
#include "rrc_api.h"
#include "rrc_fit.h"

static const char *MODEL =
    "<?xml version='1.0' encoding='UTF-8'?>"
    "<sbml xmlns='http://www.sbml.org/sbml/level3/version2/core' level='3' version='2'><model id='m'>"
    "<listOfCompartments><compartment id='c' size='1' constant='true'/></listOfCompartments>"
    "<listOfSpecies>"
    "<species id='X0' compartment='c' initialConcentration='10' hasOnlySubstanceUnits='false' boundaryCondition='true' constant='false'/>"
    "<species id='S1' compartment='c' initialConcentration='0' hasOnlySubstanceUnits='false' boundaryCondition='false' constant='false'/>"
    "</listOfSpecies>"
    "<listOfParameters><parameter id='k1' value='0.5' constant='true'/><parameter id='k2' value='0.2' constant='true'/></listOfParameters>"
    "<listOfReactions>"
    "<reaction id='J1' reversible='false'><listOfReactants><speciesReference species='X0' stoichiometry='1' constant='true'/></listOfReactants>"
    "<listOfProducts><speciesReference species='S1' stoichiometry='1' constant='true'/></listOfProducts>"
    "<kineticLaw><math xmlns='http://www.w3.org/1998/Math/MathML'><apply><times/><ci>k1</ci><ci>X0</ci></apply></math></kineticLaw></reaction>"
    "<reaction id='J2' reversible='false'><listOfReactants><speciesReference species='S1' stoichiometry='1' constant='true'/></listOfReactants>"
    "<kineticLaw><math xmlns='http://www.w3.org/1998/Math/MathML'><apply><times/><ci>k2</ci><ci>S1</ci></apply></math></kineticLaw></reaction>"
    "</listOfReactions></model></sbml>";

static int progress(int finished, int total, double best, void *user) {
    (void)user;
    printf("  start %d of %d done, best objective %.6g\n", finished, total, best);
    return 0; /* nonzero would cancel the run */
}

static int fail(const char *what) {
    fprintf(stderr, "%s failed: %s\n", what, getLastError());
    return 1;
}

int main(void) {
    const double k1 = 0.8, k2 = 0.3; /* the truth the data come from */
    double times[8], a[8], b[8];
    for (int i = 0; i < 8; i++) {
        times[i] = 1.0 + 2.0 * i;
        a[i] = k1 * 10.0 / k2 * (1.0 - exp(-k2 * times[i]));
        b[i] = k1 * 5.0 / k2 * (1.0 - exp(-k2 * times[i]));
    }

    /* the problem: from a loaded RoadRunner instance (fitCreateProblemFromSBML or fitLoadPEtab work too) */
    RRHandle rr = createRRInstance();
    if (!loadSBML(rr, MODEL)) return fail("loadSBML");
    FitProblemHandle p = fitCreateProblem(rr);
    if (!p) return fail("fitCreateProblem");
    freeRRInstance(rr);

    const char *targets[] = {"X0"}, *values[] = {"5"};
    if (!fitAddParameter(p, "k1", "log10", 1e-3, 1e3, 1.0, true)
        || !fitAddParameter(p, "k2", "log10", 1e-3, 1e3, 1.0, true)
        || !fitAddCondition(p, "full", 0, NULL, NULL)
        || !fitAddCondition(p, "half", 1, targets, values)
        || !fitAddObservable(p, "S1_obs", "S1", "0.1", "lin", "normal")
        || !fitAddMeasurements(p, "full", "S1_obs", 8, times, a)
        || !fitAddMeasurements(p, "half", "S1_obs", 8, times, b))
        return fail("building the problem");
    printf("%d free parameters, %d measurements, least squares: %d\n",
           fitGetNumFreeParameters(p), fitGetNumMeasurements(p), fitIsLeastSquares(p));

    /* the objective and its exact gradient at the nominal values */
    RRVectorPtr nominal = fitGetNominalValues(p);
    double f;
    RRVectorPtr g = fitGradient(p, nominal->Data, nominal->Count, &f);
    if (!g) return fail("fitGradient");
    printf("at the nominal values: objective %.6g, gradient (%.6g, %.6g)\n", f, g->Data[0], g->Data[1]);
    freeVector(g);
    freeVector(nominal);

    /* multistart least squares */
    FitOptionsHandle o = fitCreateOptions();
    fitSetOption(o, "n_starts", 6);
    fitSetOption(o, "seed", 42);
    FitResultHandle r = fitRun(p, o, progress, NULL);
    if (!r) return fail("fitRun");
    double best;
    RRVectorPtr x = fitGetBest(r, &best);
    RRDoubleMatrixPtr starts = fitGetStarts(r);
    printf("best objective %.10g: k1 = %.8g, k2 = %.8g (%s)\n", best, pow(10, x->Data[0]), pow(10, x->Data[1]),
           fitGetStatusMessage((int)starts->Data[2]));

    /* 95%% Fisher intervals, back on the linear scale */
    RRDoubleMatrixPtr ci = fitFisherIntervals(p, x->Data, x->Count, 0.95);
    if (!ci) return fail("fitFisherIntervals");
    RRStringArrayPtr ids = fitGetFreeParameterIds(p);
    for (int i = 0; i < ci->RSize; i++) {
        const double *row = ci->Data + 5 * i;
        printf("  %s in [%.6g, %.6g]%s\n", ids->String[i], pow(10, row[2]), pow(10, row[3]), row[4] ? "" : " (not identifiable)");
    }

    int ok = fabs(pow(10, x->Data[0]) - k1) < 1e-5 && fabs(pow(10, x->Data[1]) - k2) < 1e-5;
    printf(ok ? "recovered the true parameters\n" : "FAILED to recover the true parameters\n");
    freeStringArray(ids);
    freeMatrix(ci);
    freeMatrix(starts);
    freeVector(x);
    fitFreeResult(r);
    fitFreeOptions(o);
    fitFreeProblem(p);
    return ok ? 0 : 1;
}
