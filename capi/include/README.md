Verbatim copies of the libroadrunner C API headers from sys-bio/roadrunner (branch `develop`, `wrappers/C/`), Apache License 2.0, (C) University of Washington. Only the functions in rrc_api.h, rrc_types.h and rrc_utilities.h are exported by this library; rrc_logging_api.h, rrc_nom_api.h and rrc_libstruct_api.h are included only because rrc_api.h `#include`s them.

`rrc_fit.h` is not from libroadrunner: it is roadrunner_nitro's own parameter-estimation API (functions prefixed `fit`), exported by the same library. See `capi/examples/fit.c`.
