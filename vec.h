#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "exitmf.h"

#define EPS 1e-0

int vecread(const char *v_fpath, float **v, int *v_len);
float *vecalloc(float **v, int v_len);
float *vecncpy(float *u, float *v, int v_len);
int veccmp(float *v, float *u, int v_len);
