#ifndef TIMING_H
#define TIMING_H

#include <time.h>

typedef struct {
    struct timespec measurementStart;

    double total;
    double average;
    double minimum;
    double maximum;

    int numEvents;
} timing_t;

void initializeTiming(timing_t* timing);
void startTimingMeasurement(timing_t* timing);
void endTimingMeasurement(timing_t* timing);
void calculateAverageTime(timing_t* timing);
void mergeTiming(timing_t* accumulator, const timing_t* sample);

#endif
