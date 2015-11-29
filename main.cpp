#include <iostream>
#include <sstream>

#include "index.hpp"

using namespace math;
using namespace util;
using namespace gfx;

//#include "examples/empty_app.hpp"
//#include "examples/arcball_app.hpp"
//#include "examples/ssao_app.hpp"
//#include "examples/skydome_app.hpp"
//#include "examples/geometry_app.hpp"
//#include "examples/terrain_app.hpp"
//#include "examples/camera_app.hpp"
//#include "examples/sandbox_app.hpp"
#include "examples/euclidean_app.hpp"

IMPLEMENT_MAIN(int argc, char * argv[])
{
    ExperimentalApp app;
    app.main_loop();
    return 0;
}
