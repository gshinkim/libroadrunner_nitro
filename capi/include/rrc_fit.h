/**
 * rrc_fit.h: parameter estimation in roadrunner_nitro (stage 1), a C API beside libroadrunner's headers.
 *
 * Not part of libroadrunner: this header is roadrunner_nitro's own. libroadrunner's headers stay unchanged and
 * every function here is prefixed `fit` so it never clashes with their names.
 *
 * Conventions (as rrc_api.h):
 *  - Opaque handles: FitProblemHandle, FitOptionsHandle, FitResultHandle, each freed by its fitFree* function.
 *  - Vectors, matrices, string arrays and text returned here are owned by the caller and freed with
 *    freeVector / freeMatrix / freeStringArray / freeText (rrc_utilities.h). Matrices are row-major.
 *  - Errors: functions return false, NULL or -1; the message is in getLastError() (rrc_api.h).
 *  - Parameter vectors ("theta") hold the free (estimated) parameters, in the order fitGetFreeParameterIds gives,
 *    each on its own scale (lin, log or log10, from the parameter table).
 *  - The objective is the negative log-likelihood of the data (PEtab's convention, including the constant terms).
 *
 * Stage 1 covers: PEtab v1 core tables or a problem built here; normal and Laplace noise; lin/log/log10
 * observable transformations and parameter scales; condition overrides of parameters, initial values and
 * compartment sizes (sizes are fixed values, not estimated);
 * objective, exact gradient, residuals and residual Jacobian; trust-region reflective least squares with
 * Latin hypercube multistart; Fisher-information intervals. Later-stage features (placeholders,
 * pre-equilibration, priors, events) are refused with an error, never ignored.
 *
 * A problem is compiled on first use (any fitGet*, objective, run or interval call) and recompiled after it
 * changes. A handle may be used from one thread at a time; the library itself runs work on all cores (option
 * "threads").
 */
#ifndef rrc_fitH
#define rrc_fitH

#include "rrc_exporter.h"
#include "rrc_types.h"

#if defined(__cplusplus)
namespace rrc
{
extern "C"
{
#endif

typedef void* FitProblemHandle;  /*!< a fitting problem */
typedef void* FitOptionsHandle;  /*!< optimizer options */
typedef void* FitResultHandle;   /*!< the result of fitRun */

/* ------------------------------------------------------------------ problem */

/** Load a PEtab v1 problem from its YAML file. NULL on error. */
C_DECL_SPEC FitProblemHandle rrcCallConv fitLoadPEtab(const char* yamlPath);

/** An empty problem on the model loaded in a RoadRunner instance (its SBML as loaded, with any edits since).
 *  The problem keeps its own copy: later changes to the instance do not affect it. */
C_DECL_SPEC FitProblemHandle rrcCallConv fitCreateProblem(RRHandle rr);

/** An empty problem on an SBML model given as text. */
C_DECL_SPEC FitProblemHandle rrcCallConv fitCreateProblemFromSBML(const char* sbml);

/** Add a parameter-table row. scale: "lin", "log" or "log10" (NULL: lin). Bounds and nominal value are linear
 *  values; an estimated parameter needs finite bounds (and lowerBound > 0 on log scales), a fixed one its nominal
 *  value. An id that is a model parameter, species or compartment overrides the model's value. Other ids can be
 *  used in observable and noise formulas and in conditions. */
C_DECL_SPEC bool rrcCallConv fitAddParameter(FitProblemHandle p, const char* id, const char* scale,
                                             double lowerBound, double upperBound, double nominalValue, bool estimate);

/** Add an experimental condition: n (target, value) pairs. A target is a model parameter, a species (its
 *  initial value, in the species' units) or a compartment (its size: a number or a fixed parameter). A value is a number, a parameter-table id, or "" / "nan" (keep the
 *  model's value). n may be 0. */
C_DECL_SPEC bool rrcCallConv fitAddCondition(FitProblemHandle p, const char* id, int n, const char** targets, const char** values);

/** Add an observable. formula and noiseFormula use model symbols (species in their SBML units), parameter-table
 *  ids and time. transformation: "lin" | "log" | "log10" (NULL: lin); noiseDistribution: "normal" | "laplace"
 *  (NULL: normal). */
C_DECL_SPEC bool rrcCallConv fitAddObservable(FitProblemHandle p, const char* id, const char* formula, const char* noiseFormula,
                                              const char* transformation, const char* noiseDistribution);

/** Add n measurements of one observable in one condition at times (>= 0; INFINITY: at the steady state reached
 *  from the last finite time). */
C_DECL_SPEC bool rrcCallConv fitAddMeasurements(FitProblemHandle p, const char* conditionId, const char* observableId,
                                                int n, const double* times, const double* values);

/** As fitAddMeasurements, the simulation starting from the steady state of condition preequilibrationId (states
 *  that conditionId sets excepted). */
C_DECL_SPEC bool rrcCallConv fitAddPreequilibratedMeasurements(FitProblemHandle p, const char* preequilibrationId,
                                                               const char* conditionId, const char* observableId,
                                                               int n, const double* times, const double* values);

/** Integrator tolerances of every simulation (default rtol 1e-8, atol 1e-12). */
C_DECL_SPEC bool rrcCallConv fitSetTolerances(FitProblemHandle p, double rtol, double atol);

C_DECL_SPEC bool rrcCallConv fitFreeProblem(FitProblemHandle p);

/** Number of free parameters (compiles the problem). -1 on error. */
C_DECL_SPEC int rrcCallConv fitGetNumFreeParameters(FitProblemHandle p);

/** Ids of the free parameters, in theta order. */
C_DECL_SPEC RRStringArrayPtr rrcCallConv fitGetFreeParameterIds(FitProblemHandle p);

/** Scales of the free parameters ("lin", "log", "log10"), in theta order. */
C_DECL_SPEC RRStringArrayPtr rrcCallConv fitGetFreeParameterScales(FitProblemHandle p);

/** Lower bounds, upper bounds and nominal values of the free parameters, on their scales. */
C_DECL_SPEC RRVectorPtr rrcCallConv fitGetLowerBounds(FitProblemHandle p);
C_DECL_SPEC RRVectorPtr rrcCallConv fitGetUpperBounds(FitProblemHandle p);
C_DECL_SPEC RRVectorPtr rrcCallConv fitGetNominalValues(FitProblemHandle p);

/** 1 if the problem is a least-squares problem (normal noise, noise independent of the state and of the free
 *  parameters): residuals, fitRun and Fisher intervals apply. 0 if not, -1 on error. */
C_DECL_SPEC int rrcCallConv fitIsLeastSquares(FitProblemHandle p);

/** Number of measurements (the length of residual vectors). -1 on error. */
C_DECL_SPEC int rrcCallConv fitGetNumMeasurements(FitProblemHandle p);

/* ------------------------------------------------------------------ objective (for hosts with their own optimizer) */

/** Negative log-likelihood at theta (n values) into *value. False if a simulation fails or the likelihood is
 *  undefined there (treat as a rejected point). */
C_DECL_SPEC bool rrcCallConv fitObjective(FitProblemHandle p, const double* theta, int n, double* value);

/** Gradient of the negative log-likelihood at theta (exact, forward sensitivities); the objective value goes to
 *  *value unless value is NULL. */
C_DECL_SPEC RRVectorPtr rrcCallConv fitGradient(FitProblemHandle p, const double* theta, int n, double* value);

/** Residuals (h(y) - h(measurement)) / sigma, in measurement order (least-squares problems). */
C_DECL_SPEC RRVectorPtr rrcCallConv fitResiduals(FitProblemHandle p, const double* theta, int n);

/** Jacobian of the residuals: measurements x free parameters. */
C_DECL_SPEC RRDoubleMatrixPtr rrcCallConv fitResidualJacobian(FitProblemHandle p, const double* theta, int n);

/** Simulated observable of every measurement at theta, in measurement order. */
C_DECL_SPEC RRVectorPtr rrcCallConv fitSimulateMeasurements(FitProblemHandle p, const double* theta, int n);

/* ------------------------------------------------------------------ optimization */

/** Options with defaults: n_starts 10, seed 0, threads 0 (all cores), ftol 1e-8, xtol 1e-8, gtol 1e-8,
 *  max_nfev 0 (100 x free parameters per start). */
C_DECL_SPEC FitOptionsHandle rrcCallConv fitCreateOptions(void);

/** Set an option by name: "n_starts", "seed", "threads", "ftol", "xtol", "gtol", "max_nfev". */
C_DECL_SPEC bool rrcCallConv fitSetOption(FitOptionsHandle o, const char* name, double value);

/** Use theta (n values) as the first start; the other starts stay Latin hypercube samples. n = 0 clears it. */
C_DECL_SPEC bool rrcCallConv fitSetStartPoint(FitOptionsHandle o, const double* theta, int n);

C_DECL_SPEC bool rrcCallConv fitFreeOptions(FitOptionsHandle o);

/** Progress: called after each start finishes with (starts finished, total starts, best objective so far, user).
 *  Called from worker threads, one call at a time. Returning nonzero cancels: no new start begins, running
 *  starts stop after their current iteration, and every start keeps its best point. */
typedef int (*FitProgressCallback)(int finished, int total, double best, void* user);

/** Multistart least-squares fit (options NULL: defaults; callback NULL: none). For a given seed the result is the
 *  same, to the last bit, whatever the number of threads. */
C_DECL_SPEC FitResultHandle rrcCallConv fitRun(FitProblemHandle p, FitOptionsHandle o, FitProgressCallback callback, void* user);

/** Number of starts in a result. -1 on error. */
C_DECL_SPEC int rrcCallConv fitGetNumStarts(FitResultHandle r);

/** Best parameters (theta order, on their scales); its objective to *value unless value is NULL. */
C_DECL_SPEC RRVectorPtr rrcCallConv fitGetBest(FitResultHandle r, double* value);

/** Every start, best first (ties in start order), one row each:
 *  [start index, objective, status, function evaluations, Jacobian evaluations, iterations, theta...].
 *  A start that failed has objective +inf. Status codes: see fitGetStatusMessage. */
C_DECL_SPEC RRDoubleMatrixPtr rrcCallConv fitGetStarts(FitResultHandle r);

/** The starting points, in the row order of fitGetStarts: starts x free parameters. */
C_DECL_SPEC RRDoubleMatrixPtr rrcCallConv fitGetStartPoints(FitResultHandle r);

/** Text for a status code (static: do not free). 0 failed at the start point, 1 evaluation budget used up,
 *  2 gradient small (gtol), 3 objective change small (ftol), 4 step small (xtol), 5 ftol and xtol, 6 cancelled.
 *  Codes 2-5 mean converged. */
C_DECL_SPEC const char* rrcCallConv fitGetStatusMessage(int status);

C_DECL_SPEC bool rrcCallConv fitFreeResult(FitResultHandle r);

/* ------------------------------------------------------------------ uncertainty */

/** Fisher-information (Wald) intervals at theta for confidence level (e.g. 0.95), from the Gauss-Newton Hessian
 *  J^T J of the residuals. One row per free parameter: [theta, standard deviation, lower, upper, identifiable],
 *  on the parameter scales. A parameter along a direction of (numerically) zero curvature has identifiable 0 and
 *  infinite standard deviation and bounds. */
C_DECL_SPEC RRDoubleMatrixPtr rrcCallConv fitFisherIntervals(FitProblemHandle p, const double* theta, int n, double level);

/** Covariance (pseudo-inverse of J^T J) at theta: free parameters x free parameters, on the parameter scales. */
C_DECL_SPEC RRDoubleMatrixPtr rrcCallConv fitFisherCovariance(FitProblemHandle p, const double* theta, int n);

#if defined(__cplusplus)
}   /* extern "C" */
}   /* namespace rrc */
#endif

#endif
