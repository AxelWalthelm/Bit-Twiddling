#include "../semi_exhaustive_search_for_8bit_rev.h"
#include "BitTwiddling.h"
#include <signal.h>
#include <unistd.h>
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
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
 
    sigaction(SIGINT, &sigIntHandler, NULL);
    sigaction(SIGTERM, &sigIntHandler, NULL);


    semi_exhaustive_search_for_8bit_rev_cuda();
    //semi_exhaustive_search_for_8bit_rev();

    if (global_do_terminate)
        printf("Terminated by global_do_terminate\n");
    return 0;
}
