#include "vec.h"

int vecread(const char *v_fpath, float **v, int *v_len)
{
    FILE *v_file = fopen(v_fpath, "r");
    if (v_file == NULL) exitmf("Bad file path\n");

    int scan_rslt, i;

    scan_rslt = fscanf(v_file, "%d", v_len);
    if (scan_rslt != 1) exitmf("Bad file format for v_len\n");
    *v = (float *)malloc(sizeof(**v) * (*v_len));
    if (v == NULL) exitmf("Failed at malloc\n");
    for (i = 0; i < *v_len; i++) {
        scan_rslt = fscanf(v_file, "%f", (*v)+i);
        if (scan_rslt != 1) {
            fprintf(stderr, "Bad file format for v[]\n");
            break;
        }
    }

    fclose(v_file);
    return i;
}

float *vecncpy(float **u, float *v, int v_len)
{
    *u = (float *)malloc(sizeof(**u) * v_len);
    if (u == NULL) exitmf("Failed at malloc in vecncpy\n");
    void *rslt = memcpy(*u, v, sizeof(*v) * v_len);
    if (rslt == NULL) exitmf("Failed at memcpy in vecncpy\n");
    return *u;
}

int veccmp(float *v, float *u, int v_len)
{
    int i;
    for (i = 0; i < v_len; i++) {
        // fprintf(stderr,"%6d: calc: %f, precalc: %f\n", i, v[i], u[i]);
        if (fabs(v[i] - u[i]) > EPS) {
            return 1;
        }
    }
    return 0;
}
