#include "args.h"

void args(int argc, char *argv[], char **v_fpath, char **vc_fpath, int *b_cnt)
{
    if (argc != 4) exitmf("Bad arguments\n");
    *v_fpath = argv[1];
    *vc_fpath = argv[2];
    *b_cnt = atoi(argv[3]);
}
