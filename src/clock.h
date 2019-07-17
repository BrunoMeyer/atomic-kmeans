#include <string.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <list>
#include <sstream>
#include <string>
#include <vector>

typedef struct {
  struct timespec xadd_time1, xadd_time2;
  long long unsigned xtotal_ns;
  long xn_events;
} chronometer_t;


void chrono_reset( chronometer_t *chrono )
{
  chrono->xtotal_ns = 0;
  chrono->xn_events = 0;
}

inline void chrono_start( chronometer_t *chrono ) {
  clock_gettime(CLOCK_MONOTONIC_RAW, &(chrono->xadd_time1) );
}

inline long long unsigned  chrono_gettotal( chronometer_t *chrono ) {
  return chrono->xtotal_ns;
}

inline long long unsigned chrono_getcount( chronometer_t *chrono ) {
  return chrono->xn_events;
}

inline void chrono_stop( chronometer_t *chrono ) {
  clock_gettime(CLOCK_MONOTONIC_RAW, &(chrono->xadd_time2) );

  long long unsigned ns1 = chrono->xadd_time1.tv_sec*1000*1000*1000 + 
              chrono->xadd_time1.tv_nsec;
  long long unsigned ns2 = chrono->xadd_time2.tv_sec*1000*1000*1000 + 
              chrono->xadd_time2.tv_nsec;
  long long unsigned deltat_ns = ns2 - ns1;

  chrono->xtotal_ns += deltat_ns;
  chrono->xn_events++;
}

void chrono_reportTime( chronometer_t *chrono, char *s ) {

printf("\n%s deltaT(ns): %llu ns for %ld ops \n"
                                    "        ==> each op takes %llu ns\n",
        s, chrono->xtotal_ns, chrono->xn_events, 
                                      chrono->xtotal_ns/chrono->xn_events );
}

void chrono_report_TimeInLoop( chronometer_t *chrono, char *s, int loop_count ) {

  printf("\n%s deltaT(ns): %llu ns for %ld ops \n"
                                    "        ==> each op takes %llu ns\n",
        s, chrono->xtotal_ns, chrono->xn_events*loop_count, 
                        chrono->xtotal_ns/(chrono->xn_events*loop_count) );
}