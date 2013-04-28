#include "args.h"

void args(int argc, char *argv[], char **v_fpath, char **vc_fpath)
{
    if (argc != 3) exitmf("Bad arguments\n");
    *v_fpath = argv[1];
    *vc_fpath = argv[2];
}
