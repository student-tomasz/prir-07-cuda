#include <stdio.h>
#include <stdlib.h>
#include "args.h"
#include "exitmf.h"
#include "vec.h"

void cube(const int v_len, float *v);


int main(int argc, char *argv[])
{
    float *v, *vc;
    int v_len, vc_len;
    char *v_fpath, *vc_fpath;
    int b_cnt;

    args(argc, argv, &v_fpath, &vc_fpath, &b_cnt);

    vecread(v_fpath, &v, &v_len);
    vecread(vc_fpath, &vc, &vc_len);

    cube(v_len, v);
    if (veccmp(v, vc, v_len) != 0) exitmf("Those sequential results suck! Abort.\n");
    printf("OK\n");

    return EXIT_SUCCESS;
}

void cube(const int v_len, float *v)
{
    int i;
    for (i = 0; i < v_len; i++) {
        v[i] = v[i] * v[i] * v[i];
    }
}
