#include <float.h>

#include "ssc_timing.h"

#define CLOCK CLOCK_REALTIME

void initializeTiming(timing_t* timing) {
    timing->total = 0.0;
    timing->average = 0.0;
    timing->minimum = DBL_MAX;
    timing->maximum = 0.0;
    timing->numEvents = 0;
}

void startTimingMeasurement(timing_t* timing) {
    clock_gettime(CLOCK, &timing->measurementStart);
}

void endTimingMeasurement(timing_t* timing) {
    struct timespec* start = &timing->measurementStart;
    struct timespec end;

    clock_gettime(CLOCK, &end);

    double seconds = end.tv_sec - start->tv_sec;
    double nanoseconds = end.tv_nsec - start->tv_nsec;
    double conversionFactor = 0.000000001f;
    double duration = seconds + nanoseconds * conversionFactor;

    timing->total += duration;
    timing->numEvents += 1;

    if (duration > timing->maximum)
        timing->maximum = duration;

    if (duration < timing->minimum)
        timing->minimum = duration;
}

void calculateAverageTime(timing_t* timing) {
    timing->average = timing->total / timing->numEvents;
}

void mergeTiming(timing_t* accumulator, const timing_t* sample) {
    accumulator->total += sample->total;
    accumulator->numEvents += sample->numEvents;

    if (sample->maximum > accumulator->maximum)
        accumulator->maximum = sample->maximum;

    if (sample->minimum < accumulator->minimum)
        accumulator->minimum = sample->minimum;
}
