#include <iostream>
#include <sstream>

#include "util.hpp"
#include "string_utils.hpp"
#include "geometric.hpp"
#include "linear_algebra.hpp"
#include "math_util.hpp"
#include "circular_buffer.hpp"
#include "concurrent_queue.hpp"
#include "try_locker.hpp"
#include "running_statistics.hpp"
#include "time_keeper.hpp"
#include "human_time.hpp"
#include "signal.hpp" // todo: rename
#include "one_euro.hpp"
#include "json.hpp"
#include "geometry.hpp"
#include "pid_controller.hpp" // todo: do integration in pid class
#include "base64.hpp"
#include "dsp_filters.hpp"
#include "bit_mask.hpp"
#include "file_io.hpp"
#include "GlMesh.hpp"
#include "GlShader.hpp"
#include "GlTexture.hpp"
#include "universal_widget.hpp"
#include "arcball.hpp"
#include "sketch.hpp"
#include "glfw_app.hpp"
#include "tinyply.h"
#include "renderable_grid.hpp"
#include "procedural_sky.hpp"
#include "nvg.hpp"
#include "nanovg_gl.h"

using namespace math;
using namespace util;
using namespace tinyply;
using namespace gfx;

static const float TEXT_OFFSET_X = 3;
static const float TEXT_OFFSET_Y = 1;

struct ExperimentalApp : public GLFWApp
{
    
    Model sofaModel;
    Geometry sofaGeometry;
    
    GlTexture emptyTex;
    
    std::unique_ptr<GLTextureView> myTexture;
    std::unique_ptr<GlShader> simpleShader;
    
    UWidget rootWidget;
    
    GlCamera camera;
    Sphere cameraSphere;
    //Arcball myArcball;
    
    float2 lastCursor;
    bool isDragging = false;
    
    RenderableGrid grid;
    
    FPSCameraController cameraController;
    PreethamProceduralSky skydome;
    
    NVGcontext * nvgCtx;
    std::shared_ptr<NvgFont> sourceFont;
    
    ExperimentalApp() : GLFWApp(600, 600, "Experimental App")
    {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        try
        {
            std::ifstream ss("assets/sofa.ply", std::ios::binary);
            PlyFile file(ss);
            
            std::vector<float> verts;
            std::vector<int32_t> faces;
            std::vector<float> texCoords;
            
            uint32_t vertexCount = file.request_properties_from_element("vertex", {"x", "y", "z"}, verts);
            uint32_t numTriangles = file.request_properties_from_element("face", {"vertex_indices"}, faces, 3);
            uint32_t uvCount = file.request_properties_from_element("face", {"texcoord"}, texCoords, 6);
            
            file.read(ss);
            
            sofaGeometry.vertices.reserve(vertexCount);
            for (int i = 0; i < vertexCount * 3; i+=3)
                sofaGeometry.vertices.push_back(math::float3(verts[i], verts[i+1], verts[i+2]));
            
            sofaGeometry.faces.reserve(numTriangles);
            for (int i = 0; i < numTriangles * 3; i+=3)
                sofaGeometry.faces.push_back(math::uint3(faces[i], faces[i+1], faces[i+2]));
            
            sofaGeometry.texCoords.reserve(uvCount);
            for (int i = 0; i < uvCount * 6; i+= 2)
                sofaGeometry.texCoords.push_back(math::float2(texCoords[i], texCoords[i+1]));

            sofaGeometry.compute_normals();
            sofaGeometry.compute_bounds();
            sofaGeometry.compute_tangents();
            
            std::cout << "Read " << vertexCount << " vertices..." << std::endl;
            
        }
        catch (std::exception e)
        {
            std::cerr << "Caught exception: " << e.what() << std::endl;
        }
        
        sofaModel.mesh = make_mesh_from_geometry(sofaGeometry);
        sofaModel.bounds = sofaGeometry.compute_bounds();
        
        gfx::gl_check_error(__FILE__, __LINE__);
        
        simpleShader.reset(new gfx::GlShader(read_file_text("assets/simple.vert"), read_file_text("assets/simple.frag")));
        
        //std::vector<uint8_t> whitePixel = {255,255,255,255};
        //emptyTex.load_data(1, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, whitePixel.data());
        
        emptyTex = load_image("assets/anvil.png");
        
        rootWidget.bounds = {0, 0, (float) width, (float) height};
        rootWidget.add_child( {{0,+10},{0,+10},{0.5,0},{0.5,0}}, std::make_shared<UWidget>());
        //rootWidget.add_child( {{0, 0}, {0.5, +10}, {0.5, 0}, {1.0, -10}}, std::make_shared<UWidget>());
        
        rootWidget.layout();
    
        myTexture.reset(new GLTextureView(emptyTex.get_gl_handle()));
        
        cameraController.set_camera(&camera);
        camera.fov = 75;
        
        grid = RenderableGrid(1, 100, 100);
        
        nvgCtx = make_nanovg_context(NVG_ANTIALIAS | NVG_STENCIL_STROKES);
        
        if(!nvgCtx)
            throw std::runtime_error("error initializing nanovg context");
        
        sourceFont = std::make_shared<NvgFont>(nvgCtx, "souce_sans_pro", read_file_binary("assets/source_code_pro_regular.ttf"));

        //cameraSphere = Sphere(sofaModel.bounds.center(), 1);
        //myArcball = Arcball(&camera, cameraSphere);
        //myArcball.set_constraint_axis(float3(0, 1, 0));
        
    }
    
    void on_window_resize(math::int2 size) override
    {
        rootWidget.bounds = {0, 0, (float) size.x, (float) size.y};
        rootWidget.layout();
    }
    
    void on_input(const InputEvent & event) override
    {
        if (event.type == InputEvent::KEY)
        {
            if (event.value[0] == GLFW_KEY_RIGHT && event.action == GLFW_RELEASE)
            {
                //sunPhi += 5;
            }
            if (event.value[0] == GLFW_KEY_LEFT && event.action == GLFW_RELEASE)
            {
                //sunPhi -= 5;
            }
            if (event.value[0] == GLFW_KEY_UP && event.action == GLFW_RELEASE)
            {
                //sunTheta += 5;
            }
            if (event.value[0] == GLFW_KEY_DOWN && event.action == GLFW_RELEASE)
            {
                //sunTheta -= 5;
            }
            if (event.value[0] == GLFW_KEY_EQUAL && event.action == GLFW_REPEAT)
            {
                camera.fov += 1;
                std::cout << camera.fov << std::endl;
            }
        }
        if (event.type == InputEvent::CURSOR && isDragging)
        {
            //if (event.cursor != lastCursor)
               // myArcball.mouse_drag(event.cursor, event.windowSize);
        }
        
        if (event.type == InputEvent::MOUSE)
        {
            if (event.is_mouse_down())
            {
                isDragging = true;
                //myArcball.mouse_down(event.cursor, event.windowSize);
            }
            
            if (event.is_mouse_up())
            {
                isDragging = false;
                //myArcball.mouse_down(event.cursor, event.windowSize);
            }
        }
        
        cameraController.handle_input(event);
        lastCursor = event.cursor;
    }
    
    void on_update(const UpdateEvent & e) override
    {
        cameraController.update(e.elapsed_s / 1000);
    }
    
    void draw_ui()
    {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        nvgBeginFrame(nvgCtx, width, height, 1.0);
        {
            nvgBeginPath(nvgCtx);
            nvgRect(nvgCtx, 10, 10, 200, 150);
            nvgStrokeColor(nvgCtx, nvgRGBA(255, 255, 255, 127));
            nvgStrokeWidth(nvgCtx, 2.0f);
            nvgStroke(nvgCtx);
            
            for (auto widget : rootWidget.children)
            {
                nvgBeginPath(nvgCtx);
                nvgRect(nvgCtx, widget->bounds.x0, widget->bounds.y0, widget->bounds.width(), widget->bounds.height());
                std::cout << widget->bounds.width() << std::endl;
                nvgStrokeColor(nvgCtx, nvgRGBA(255, 255, 255, 255));
                nvgStrokeWidth(nvgCtx, 1.0f);
                nvgStroke(nvgCtx);
            }
            
            std::string text = "Hello NanoVG";
            const float textX = 15 + TEXT_OFFSET_X, textY = 15 + TEXT_OFFSET_Y;
            nvgFontFaceId(nvgCtx, sourceFont->id);
            nvgFontSize(nvgCtx, 20);
            nvgTextAlign(nvgCtx, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
            nvgBeginPath(nvgCtx);
            nvgFillColor(nvgCtx, nvgRGBA(0,0,0,255));
            auto ret = nvgText(nvgCtx, textX, textY, text.c_str(), nullptr);
            
        }
        nvgEndFrame(nvgCtx);
    }
    
    void on_draw() override
    {
        static int frameCount = 0;
        
        glfwMakeContextCurrent(window);
        
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        
        glDisable(GL_POLYGON_OFFSET_FILL);
        
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
     
        const auto proj = camera.get_projection_matrix((float) width / (float) height);
        const float4x4 view = camera.get_view_matrix();
        const float4x4 viewProj = mul(proj, view);
        
        skydome.render(viewProj, camera.get_eye_point(), camera.farClip);
        
        {
            simpleShader->bind();
            
            simpleShader->uniform("u_viewProj", viewProj);
            simpleShader->uniform("u_eye", float3(0, 10, -10));
            
            simpleShader->uniform("u_emissive", float3(.33f, 0.36f, 0.275f));
            simpleShader->uniform("u_diffuse", float3(0.2f, 0.4f, 0.25f));
            
            simpleShader->uniform("u_lights[0].position", float3(5, 10, -5));
            simpleShader->uniform("u_lights[0].color", float3(0.7f, 0.2f, 0.2f));
            
            simpleShader->uniform("u_lights[1].position", float3(-5, 10, 5));
            simpleShader->uniform("u_lights[1].color", float3(0.4f, 0.8f, 0.4f));
            
            {
                sofaModel.pose.position = float3(0, -1, -4);
                //sofaModel.pose.orientation = qmul(myArcball.get_quat(), sofaModel.pose.orientation);
                
                //std::cout <<  sofaModel.pose.orientation << std::endl;
                
                auto model = mul(sofaModel.pose.matrix(), make_scaling_matrix(0.001));
                
                simpleShader->uniform("u_modelMatrix", model);
                simpleShader->uniform("u_modelMatrixIT", inv(transpose(model)));
                sofaModel.draw();
            }
            
            {
                auto model = make_scaling_matrix(1);
                simpleShader->uniform("u_modelMatrix", model);
                simpleShader->uniform("u_modelMatrixIT", inv(transpose(model)));
                //skyMesh.draw_elements();
            }
            
            simpleShader->unbind();
        }
        
        grid.render(proj, view);
        
        gfx::gl_check_error(__FILE__, __LINE__);
        
        for (auto widget : rootWidget.children)
        {
            myTexture->draw(widget->bounds, int2(width, height));
        }

        gfx::gl_check_error(__FILE__, __LINE__);
        
        draw_ui();
        
        glfwSwapBuffers(window);
        
        frameCount++;
    }
    
};

IMPLEMENT_MAIN(int argc, char * argv[])
{
    ExperimentalApp app;
    app.main_loop();
    return 0;
}

