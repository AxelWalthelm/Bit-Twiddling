#include "../semi_exhaustive_search_for_8bit_rev.h"
#include "BitTwiddling.h"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>

std::atomic<bool> global_do_terminate(false);

static
void my_handler(int s)
{
    printf("Caught signal %d with global_do_terminate %d\n", s, (bool)global_do_terminate);
    if (!global_do_terminate)
    {
        global_do_terminate = true;
    }
    else
    {
        printf("Terminated by signal - results may be corrupted\n");
        exit(1);
    }
}

int main()
{
    signal(SIGINT, my_handler);

    semi_exhaustive_search_for_8bit_rev_cuda();
    //semi_exhaustive_search_for_8bit_rev();

    if (global_do_terminate)
        printf("Terminated by global_do_terminate\n");
    return 0;
}
