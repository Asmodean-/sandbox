#include <iostream>

#include "geometric.hpp"
#include "linear_algebra.hpp"
#include "math_util.hpp"
#include "circular_buffer.hpp"
#include "concurrent_queue.hpp"
#include "try_locker.hpp"
#include "running_statistics.hpp"

using namespace math;

int main(int argc, const char * argv[])
{
    ConcurrentQueue<float> queue;
    RunningStats<float> stats;
    return 0;
}
