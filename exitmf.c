#include "exitmf.h"

void exitmf(const char *msg)
{
    fprintf(stderr, msg);
    exit(EXIT_FAILURE);
}
