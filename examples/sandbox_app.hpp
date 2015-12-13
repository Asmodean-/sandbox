#include "index.hpp"

using namespace math;
using namespace util;
using namespace gfx;

struct ExperimentalApp : public GLFWApp
{
    uint64_t frameCount = 0;

    GlCamera camera;
    HosekProceduralSky skydome;
    RenderableGrid grid;
    FlyCameraController cameraController;
    
    std::vector<Renderable> proceduralModels;
    std::vector<Renderable> cameraPositions;
    std::vector<LightObject> lights;
    
    std::unique_ptr<GlShader> simpleShader;
    
    std::unique_ptr<MeshLine> meshline;
    
    std::random_device rd;
    std::mt19937 gen;
    
    ExperimentalApp() : GLFWApp(940, 720, "Sandbox App")
    {
        gen = std::mt19937(rd());
    
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        cameraController.set_camera(&camera);
        
        camera.look_at({0, 8, 24}, {0, 0, 0});
        
        meshline.reset(new MeshLine(camera, float2(width, height), 1.0f, float3(1.0, 1.0, 1.0)));
        auto newSpline = create_curve();
        meshline->set_vertices(newSpline);
        
        simpleShader.reset(new gfx::GlShader(read_file_text("assets/shaders/simple_vert.glsl"), read_file_text("assets/shaders/simple_frag.glsl")));
        
        {
            lights.resize(2);
            lights[0].color = float3(249.f / 255.f, 228.f / 255.f, 157.f / 255.f);
            lights[0].pose.position = float3(25, 15, 0);
            lights[1].color = float3(255.f / 255.f, 242.f / 255.f, 254.f / 255.f);
            lights[1].pose.position = float3(-25, 15, 0);
        }
        
        {
            proceduralModels.resize(4);
            
            proceduralModels[0] = Renderable(make_sphere(1.0));
            proceduralModels[0].pose.position = float3(0, 2, +8);
            
            proceduralModels[1] = Renderable(make_cube());
            proceduralModels[1].pose.position = float3(0, 2, -8);
            
            proceduralModels[2] = Renderable(make_icosahedron());
            proceduralModels[2].pose.position = float3(8, 2, 0);
            
            proceduralModels[3] = Renderable(make_octohedron());
            proceduralModels[3].pose.position = float3(-8, 2, 0);
        }
        
        grid = RenderableGrid(1, 64, 64);
        
        gfx::gl_check_error(__FILE__, __LINE__);
    }
    
    
    std::vector<float3> create_curve(float rMin = 3.f, float rMax = 12.f)
    {
        std::vector<float3> curve;
        
        auto r = std::uniform_real_distribution<float>(0.0, 1.0);
        
        ConstantSpline s;
        
        s.p0 = float3(0, 0, 0);
        s.p1 = s.p0 + float3( .5f - r(gen), .5f - r(gen), .5f - r(gen));
        s.p2 = s.p1 + float3( .5f - r(gen), .5f - r(gen), .5f - r(gen));
        s.p3 = s.p2 + float3( .5f - r(gen), .5f - r(gen), .5f - r(gen));
        
        s.p0 *= rMin + r(gen) * rMax;
        s.p1 *= rMin + r(gen) * rMax;
        s.p2 *= rMin + r(gen) * rMax;
        s.p3 *= rMin + r(gen) * rMax;
        
        s.calculate(.001f);
        s.calculate_distances();
        s.reticulate(256);
        
        auto sPoints = s.get_spline();
        
        for (const auto & p : sPoints)
        {
            curve.push_back(p);
            curve.push_back(p);
        }
        
        return curve;
    }
    
    void on_window_resize(math::int2 size) override
    {

    }
    
    void on_input(const InputEvent & event) override
    {
        cameraController.handle_input(event);
    }
    
    void on_update(const UpdateEvent & e) override
    {
        cameraController.update(e.timestep_ms);
    }
    
    void on_draw() override
    {
        glfwMakeContextCurrent(window);
        
        glEnable(GL_CULL_FACE);
        glEnable(GL_DEPTH_TEST);

        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
     
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

        const auto proj = camera.get_projection_matrix((float) width / (float) height);
        const float4x4 view = camera.get_view_matrix();
        const float4x4 viewProj = mul(proj, view);
        
        skydome.render(viewProj, camera.get_eye_point(), camera.farClip);
        
        // Simple Shader
        {
            simpleShader->bind();
            
            simpleShader->uniform("u_viewProj", viewProj);
            simpleShader->uniform("u_eye", camera.get_eye_point());
            
            simpleShader->uniform("u_emissive", float3(.10f, 0.10f, 0.10f));
            simpleShader->uniform("u_diffuse", float3(0.4f, 0.4f, 0.4f));
            
            for (int i = 0; i < lights.size(); i++)
            {
                auto light = lights[i];
                
                simpleShader->uniform("u_lights[" + std::to_string(i) + "].position", light.pose.position);
                simpleShader->uniform("u_lights[" + std::to_string(i) + "].color", light.color);
            }
            
            for (const auto & model : proceduralModels)
            {
                simpleShader->uniform("u_modelMatrix", model.get_model());
                simpleShader->uniform("u_modelMatrixIT", inv(transpose(model.get_model())));
                model.draw();
            }
            gfx::gl_check_error(__FILE__, __LINE__);
            
            simpleShader->unbind();
        }
        
        meshline->draw();
        
        grid.render(proj, view);

        gfx::gl_check_error(__FILE__, __LINE__);
        
        glfwSwapBuffers(window);
        
        frameCount++;
    }
    
};
