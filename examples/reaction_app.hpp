#include "index.hpp"

struct ExperimentalApp : public GLFWApp
{
    uint64_t frameCount = 0;
    
    GlCamera camera;
    FlyCameraController cameraController;
    
    GlMesh fullscreen_reaction_quad;
    
    ShaderMonitor shaderMonitor;
    
    std::random_device rd;
    std::mt19937 gen;
    
    UIComponent rootWidget;
    
    GlTexture gsOutput;
    std::unique_ptr<GLTextureView> gsOutputView;
    
    std::unique_ptr<GrayScottSimulator> gs;
    
    std::vector<uint8_t> pixels;
    
    float frameDelta = 0.0f;
    
    std::shared_ptr<GlShader> displacementShader;
    Renderable displacementMesh;
    
    ExperimentalApp() : GLFWApp(1280, 720, "Gray-Scott Reaction-Diffusion Simulation")
    {
        gen = std::mt19937(rd());
        
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        cameraController.set_camera(&camera);
        
        camera.look_at({-5, 15, 0}, {0, 0, 0});
        
        pixels.resize(256 * 256 * 3, 150);
        gs.reset(new GrayScottSimulator(float2(256, 256), false));
        gs->set_coefficients(0.023f, 0.077f, 0.16f, 0.08f);
        
        fullscreen_reaction_quad = make_fullscreen_quad();
        
        displacementMesh = Renderable(make_plane(48, 48, 256, 256));
        
        gsOutput.load_data(256, 256, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        gsOutputView.reset(new GLTextureView(gsOutput.get_gl_handle()));
        
        displacementShader.reset(new GlShader(read_file_text("assets/shaders/displacement_vert.glsl"), read_file_text("assets/shaders/displacement_frag.glsl")));
        shaderMonitor.add_shader(displacementShader, "assets/shaders/displacement_vert.glsl", "assets/shaders/displacement_frag.glsl");
        
        rootWidget.bounds = {0, 0, (float) width, (float) height};
        rootWidget.add_child( {{0.f,+10.f},{0.f,+10.f},{0.15f,0},{0.15f,0}}, std::make_shared<UIComponent>());
        rootWidget.layout();
        
        gl_check_error(__FILE__, __LINE__);
    }
    
    void on_window_resize(int2 size) override
    {
        
    }
    
    void on_input(const InputEvent & event) override
    {
        if (event.type == InputEvent::Type::KEY)
        {
            if (event.value[0] == GLFW_KEY_SPACE)
            {
                gs->reset();
            }
        }
        auto rX = remap<float>(event.cursor.x, 0, event.windowSize.x, 0, 256, true);
        auto rY = remap<float>(event.cursor.y, 0, event.windowSize.y, 0, 256, true);
        gs->trigger_region(rX , 256 - rY, 10, 10);
        cameraController.handle_input(event);
        
    }
    
    void on_update(const UpdateEvent & e) override
    {
        frameDelta = e.timestep_ms * 1000;
        cameraController.update(e.timestep_ms);
        shaderMonitor.handle_recompile();
    }
    
    void draw_ui()
    {
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        gsOutputView->draw(rootWidget.children[0]->bounds, int2(width, height));
    }
    
    void on_draw() override
    {
        glfwMakeContextCurrent(window);
        
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        
        const float4x4 model = make_rotation_matrix({1, 0, 0}, ANVIL_PI / 2);;
        const float4x4 viewProj = camera.get_projection_matrix((float) width / (float) height) * camera.get_view_matrix();
        
        for (int i = 0; i < 4; ++i) gs->update(frameDelta);
        
        auto output = gs->output_v();
        double cellValue;
        for (int i = 0; i < output.size(); ++i)
        {
            cellValue = output[i];
            uint32_t col = 255 - (uint32_t)(std::min<uint32_t>(255, cellValue * 768));
            pixels[i * 3 + 0] = col;
            pixels[i * 3 + 1] = col;
            pixels[i * 3 + 2] = col;
        }
        
        gsOutput.load_data(256, 256, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        
        {
            displacementShader->bind();
            
            displacementShader->uniform("u_modelMatrix", model);
            displacementShader->uniform("u_modelMatrixIT", inv(transpose(model)));
            displacementShader->uniform("u_viewProj", viewProj);
            displacementShader->texture("u_displacementTex", 0, gsOutput);
            
            displacementShader->uniform("u_eye", camera.get_eye_point());
            displacementShader->uniform("u_diffuse", float3(0.4f, 0.4f, 0.4f));
            displacementShader->uniform("u_emissive", float3(.10f, 0.10f, 0.10f));
            
            displacementMesh.draw();

            displacementShader->unbind();
        }
        
        draw_ui();
        
        gl_check_error(__FILE__, __LINE__);
        
        glfwSwapBuffers(window);
        
        frameCount++;
    }
    
};
