#include "index.hpp"

// http://developer.download.nvidia.com/presentations/2008/GDC/GDC08_SoftShadowMapping.pdf
// https://mynameismjp.wordpress.com/2015/02/18/shadow-sample-update/
// https://github.com/NVIDIAGameWorks/OpenGLSamples/blob/master/samples/gl4-maxwell/CascadedShadowMapping/CascadedShadowMappingRenderer.cpp
// https://blogs.aerys.in/jeanmarc-leroux/2015/01/21/exponential-cascaded-shadow-mapping-with-webgl/

// [ ] Stencil Reflections + Shadows
// [ ] Shadow Volumes (face / edge)
// [ ] Simple Shadow Mapping (SSM)
// [ ] Variance Shadow Mapping (VSM) http://www.punkuser.net/vsm/vsm_paper.pdf
// [ ] Exponential Shadow Mapping (ESM)
// [ ] Cascaded Shadow Mapping (CSM)
// [ ] Percentage Closer Filtering (PCF) + poisson disk sampling (PCSS + PCF)
// [ ] Moment Shadow Mapping [MSM]

inline float mix(float a, float b, float t)
{
    return a * (1 - t) + b * t;
}

std::shared_ptr<GlShader> make_watched_shader(ShaderMonitor & mon, const std::string vertexPath, const std::string fragPath, const std::string geomPath = "")
{
    std::shared_ptr<GlShader> shader = std::make_shared<GlShader>(read_file_text(vertexPath), read_file_text(fragPath), read_file_text(geomPath));
    mon.add_shader(shader, vertexPath, fragPath);
    return shader;
}

inline std::array<float3, 4> make_near_clip_coords(GlCamera & cam, float nearClip, float aspectRatio)
{
    float3 viewDirection = normalize(cam.get_view_direction());
    float3 eye = cam.get_eye_point();
    
    auto mU = transform_vector(cam.pose.orientation, float3(1, 0, 0));
    auto mV = transform_vector(cam.pose.orientation, float3(0, 1, 0));
    
    auto coords = cam.make_frustum_coords(aspectRatio); // top, right, bottom, left
    
    transform_vector(cam.pose.orientation, float3(1, 0, 0));
    
    float3 topLeft = eye + (nearClip * viewDirection) + (coords[0] * mV) + (coords[3] * mU);
    float3 topRight = eye + (nearClip * viewDirection) + (coords[0] * mV) + (coords[1] * mU);
    float3 bottomLeft = eye + (nearClip * viewDirection) + (coords[2] * mV) + (coords[3] * mU);
    float3 bottomRight = eye + (nearClip * viewDirection) + (coords[2] * mV) + (coords[1] * mU);
    
    return {topLeft, topRight, bottomLeft, bottomRight};
}

inline std::array<float3, 4> make_far_clip_coords(GlCamera & cam, float nearClip, float farClip, float aspectRatio)
{
    float3 viewDirection = normalize(cam.get_view_direction());
    float ratio = farClip / nearClip;
    float3 eye = cam.get_eye_point();
    
    auto mU = transform_vector(cam.pose.orientation, float3(1, 0, 0));
    auto mV = transform_vector(cam.pose.orientation, float3(0, 1, 0));
    
    auto coords = cam.make_frustum_coords(aspectRatio); // top, right, bottom, left
    
    float3 topLeft = eye + (farClip * viewDirection) + (ratio * coords[0] * mV) + (ratio * coords[3] * mU);
    float3 topRight = eye + (farClip* viewDirection) + (ratio * coords[0] * mV) + (ratio * coords[1] * mU);
    float3 bottomLeft = eye + (farClip * viewDirection) + (ratio * coords[2] * mV) + (ratio * coords[3] * mU);
    float3 bottomRight = eye + (farClip * viewDirection) + (ratio * coords[2] * mV) + (ratio * coords[1] * mU);
    
    return {topLeft, topRight, bottomLeft, bottomRight};
}

struct ShadowCascade
{
    
    GlTexture3D shadowArrayColor;
    GlTexture3D shadowArrayDepth;
    GlFramebuffer shadowArrayFramebuffer;
    
    GlTexture3D blurTexAttach;
    GlFramebuffer blurFramebuffer;
    
    std::vector<float4x4> viewMatrices;
    std::vector<float4x4> projMatrices;
    std::vector<float4x4> shadowMatrices;
    
    std::vector<float2> splitPlanes;
    std::vector<float> nearPlanes;
    std::vector<float> farPlanes;
    
    float resolution = 1024.f; // shadowmap resolution
    float expCascade = 120.f; // overshadowing constant
    float splitLambda = 0.5f; // frustum split constant
    
    GlShader * debugProg;
    GlShader * filterProg;
    
    GlMesh fullscreen_post_quad;
    
    ShadowCascade(GlShader * debugProg, GlShader * filterProg) : debugProg(debugProg), filterProg(filterProg)
    {
        create_framebuffers();
        fullscreen_post_quad = make_fullscreen_quad();
    }
    
    void update(GlCamera & sceneCamera, const float3 lightDir, float aspectRatio)
    {
        viewMatrices.clear();
        projMatrices.clear();
        shadowMatrices.clear();
        splitPlanes.clear();
        nearPlanes.clear();
        farPlanes.clear();
        
        float near = sceneCamera.nearClip;
        float far = sceneCamera.farClip;
        
        for (size_t i = 0; i < 4; ++i)
        {
            // Find the split planes using GPU Gem 3. Chap 10 "Practical Split Scheme".
            float splitNear = i > 0 ? mix(near + (static_cast<float>(i) / 4.0f) * (far - near), near * pow(far / near, static_cast<float>(i) / 4.0f), splitLambda) : near;
            float splitFar = i < 4 - 1 ? mix(near + (static_cast<float>(i + 1) / 4.0f) * (far - near), near * pow(far / near, static_cast<float>(i + 1) / 4.0f), splitLambda) : far;
            
            auto nc = make_near_clip_coords(sceneCamera, splitNear, aspectRatio);
            auto fc = make_far_clip_coords(sceneCamera, splitNear, splitFar, aspectRatio);
            
            float4 splitVertices[8] = { float4(nc[0], 1.0f), float4(nc[1], 1.0f), float4(nc[2], 1.0f), float4(nc[3], 1.0f), float4(fc[0], 1.0f), float4(fc[1], 1.0f), float4(fc[2], 1.0f), float4(fc[3], 1.0f) };
            
            // Split centroid for the view matrix
            float4 splitCentroid = {0, 0, 0, 0};
            for(size_t i = 0; i < 8; ++i)
            {
                splitCentroid += splitVertices[i];
            }
            splitCentroid /= 8.0f;
            
            // Make the view matrix
            float dist = max(splitFar - splitNear, distance(fc[0], fc[1]));
            auto pose = look_at_pose(splitCentroid.xyz() - lightDir * dist, splitCentroid.xyz());
            float4x4 viewMat = pose.matrix();

            // Xform split vertices to the light view space
            float4 splitVerticesLS[8];
            for (size_t i = 0; i < 8; ++i)
            {
                splitVerticesLS[i] = viewMat * splitVertices[i]; // ???
            }
            
            // Find the frustum bounding box in viewspace
            float4 min = splitVerticesLS[0];
            float4 max = splitVerticesLS[0];
            for (size_t i = 1; i < 8; ++i)
            {
                min = avl::min(min, splitVerticesLS[i]);
                max = avl::max(max, splitVerticesLS[i]);
            }
            
            // Ortho projection matrix with the corners
            float nearOffset = 10.0f;
            float farOffset = 20.0f;
            float4x4 projMat = make_orthographic_matrix(min.x, max.x, min.y, max.y, -max.z - nearOffset, -min.z + farOffset);
            const float4x4 offsetMat = float4x4(float4(0.5f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.5f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.5f, 0.0f), float4(0.5f, 0.5f, 0.5f, 1.0f)); // fixme
            
            viewMatrices.push_back(viewMat);
            projMatrices.push_back(projMat);
            shadowMatrices.push_back(offsetMat * projMat * viewMat); // order
            splitPlanes.push_back(float2(splitNear, splitFar));
            nearPlanes.push_back(-max.z - nearOffset);
            farPlanes.push_back(-min.z + farOffset);
        }
    }
    
    void filter(float2 screen)
    {
        blurFramebuffer.bind_to_draw();
        
        glDisable(GL_ALPHA_TEST);
        glDisable(GL_BLEND);
        glViewport(0, 0, resolution, resolution);

        glClearColor(0.0f, 0.00f, 0.00f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        filterProg->bind();
        
        // Configured for a 5x5
        filterProg->uniform("blurSize", float2(1.0f) / float2(resolution, resolution));
        filterProg->uniform("sigma", 3.0f);

        // Horizontal
        filterProg->texture("blurSampler", 0, GL_TEXTURE_2D_ARRAY, shadowArrayColor);
        filterProg->uniform("numBlurPixelsPerSide", 2.0f);
        filterProg->uniform("blurMultiplyVec", float2(1.0f, 0.0f));
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        fullscreen_post_quad.draw_elements();
        
        // Vertical
        filterProg->texture("blurSampler", 0, GL_TEXTURE_2D_ARRAY, blurTexAttach);
        filterProg->uniform("numBlurPixelsPerSide", 2.0f);
        filterProg->uniform("blurMultiplyVec", float2(0.0f, 1.0f));
        glDrawBuffer(GL_COLOR_ATTACHMENT1);
        fullscreen_post_quad.draw_elements();
        
        filterProg->unbind();
        
        // Reset state
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        glViewport(0, 0, screen.x, screen.y);
        
        gl_check_error(__FILE__, __LINE__);
    }
    
    void create_framebuffers()
    {
        shadowArrayColor.load_data(resolution, resolution, 4, GL_TEXTURE_2D_ARRAY, GL_R16F, GL_RGB, GL_FLOAT, nullptr);
        shadowArrayDepth.load_data(resolution, resolution, 4, GL_TEXTURE_2D_ARRAY, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        shadowArrayFramebuffer.attach(GL_COLOR_ATTACHMENT0, shadowArrayColor);
        shadowArrayFramebuffer.attach(GL_DEPTH_ATTACHMENT, shadowArrayDepth);
        if (!shadowArrayFramebuffer.check_complete()) throw std::runtime_error("incomplete shadow framebuffer");
        
        gl_check_error(__FILE__, __LINE__);
        
        blurTexAttach.load_data(resolution, resolution, 4, GL_TEXTURE_2D_ARRAY, GL_R16F, GL_RGB, GL_FLOAT, nullptr);
        blurFramebuffer.attach(GL_COLOR_ATTACHMENT0, blurTexAttach);
        blurFramebuffer.attach(GL_COLOR_ATTACHMENT1, shadowArrayColor); // note this attach point
        if (!blurFramebuffer.check_complete()) throw std::runtime_error("incomplete blur framebuffer");
        
        gl_check_error(__FILE__, __LINE__);
    };
    
};

struct ExperimentalApp : public GLFWApp
{
    uint64_t frameCount = 0;

    GlCamera camera;
    PreethamProceduralSky skydome;
    FlyCameraController cameraController;
    ShaderMonitor shaderMonitor;
    Space uiSurface;
    
    std::unique_ptr<ShadowCascade> cascade;
    
    std::unique_ptr<gui::ImGuiManager> igm;
    
    std::shared_ptr<GLTextureView> viewA;
    std::shared_ptr<GLTextureView> viewB;
    
    std::vector<Renderable> sceneObjects;
    std::vector<LightObject> lights;
    
    std::shared_ptr<GlShader> objectShader;
    std::shared_ptr<GlShader> gaussianBlurShader;
    std::shared_ptr<GlShader> shadowDebugShader;
    std::shared_ptr<GlShader> shadowCascadeShader;
    std::shared_ptr<GlShader> sceneCascadeShader;
    
    float3 lightDir = {-1.4f, -0.37f, 0.63f};
    bool showCascades = false;
    
    ExperimentalApp() : GLFWApp(1280, 720, "Shadow Mapping App")
    {
        glfwSwapInterval(0);
        
        igm.reset(new gui::ImGuiManager(window));
        gui::make_dark_theme();
        
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        cameraController.set_camera(&camera);
        camera.farClip = 80.f;
        camera.look_at({0, 0, +50}, {0, 0, 0});
        
        // Debugging views
        uiSurface.bounds = {0, 0, (float) width, (float) height};
        uiSurface.add_child( {{0.0000f, +10},{0, +10},{0.1667f, -10},{0.133f, +10}});
        uiSurface.add_child( {{0.1667f, +10},{0, +10},{0.3334f, -10},{0.133f, +10}});
        uiSurface.add_child( {{0.3334f, +10},{0, +10},{0.5009f, -10},{0.133f, +10}});
        uiSurface.add_child( {{0.5000f, +10},{0, +10},{0.6668f, -10},{0.133f, +10}});
        uiSurface.add_child( {{0.6668f, +10},{0, +10},{0.8335f, -10},{0.133f, +10}});
        uiSurface.add_child( {{0.8335f, +10},{0, +10},{1.0000f, -10},{0.133f, +10}});
        uiSurface.layout();
        
        //viewA.reset(new GLTextureView(tex.get_gl_handle()));
        //viewB.reset(new GLTextureView(tex.get_gl_handle()));
        
        objectShader = make_watched_shader(shaderMonitor, "assets/shaders/simple_vert.glsl", "assets/shaders/simple_frag.glsl");
        gaussianBlurShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/gaussian_blur_vert.glsl", "assets/shaders/shadow/gaussian_blur_frag.glsl");
        shadowDebugShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/debug_vert.glsl", "assets/shaders/shadow/debug_frag.glsl");
        shadowCascadeShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/shadowcascade_vert.glsl", "assets/shaders/shadow/shadowcascade_frag.glsl", "assets/shaders/shadow/shadowcascade_geom.glsl");
        sceneCascadeShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/cascade_vert.glsl", "assets/shaders/shadow/cascade_frag.glsl");
        
        cascade.reset(new ShadowCascade(shadowCascadeShader.get(), gaussianBlurShader.get()));
                      
        lights.resize(2);
        lights[0].color = float3(249.f / 255.f, 228.f / 255.f, 157.f / 255.f);
        lights[0].pose.position = float3(25, 15, 0);
        lights[1].color = float3(255.f / 255.f, 242.f / 255.f, 254.f / 255.f);
        lights[1].pose.position = float3(-25, 15, 0);
        
        auto hollowCube = load_geometry_from_ply("assets/models/geometry/CubeHollowOpen.ply");
        for (auto & v : hollowCube.vertices) v *= 0.20f;
        sceneObjects.push_back(Renderable(hollowCube));
        sceneObjects.back().pose.position = float3(0, 0, 0);
        sceneObjects.back().pose.orientation = make_rotation_quat_around_x(ANVIL_PI / 2);
        
        auto torusKnot = load_geometry_from_ply("assets/models/geometry/TorusKnotUniform.ply");
        for (auto & v : torusKnot.vertices) v *= 0.095f;
        sceneObjects.push_back(Renderable(torusKnot));
        sceneObjects.back().pose.position = float3(0, 0, 0);
        
        gl_check_error(__FILE__, __LINE__);
    }
    
    void on_window_resize(int2 size) override
    {

    }
    
    void on_input(const InputEvent & e) override
    {
        if (igm) igm->update_input(e);
        cameraController.handle_input(e);
    }
    
    void on_update(const UpdateEvent & e) override
    {
        cameraController.update(e.timestep_ms);
        shaderMonitor.handle_recompile();
    }
    
    void on_draw() override
    {
        //auto lightDir = skydome.get_light_direction();
        //auto sunDir = skydome.get_sun_direction();
        //auto sunPosition = skydome.get_sun_position();
        
        glfwMakeContextCurrent(window);
        
        if (igm) igm->begin_frame();

        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
     
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        

        const auto proj = camera.get_projection_matrix((float) width / (float) height);
        const float4x4 view = camera.get_view_matrix();
        const float4x4 viewProj = mul(proj, view);
        
        // Recreate cascades from camera view
        cascade->update(camera, lightDir, ((float) width / (float) height));
        
        // Render shadowmaps

        {
            auto & shadowFbo = cascade->shadowArrayFramebuffer;
            shadowFbo.bind_to_draw();
            
            glEnable(GL_CULL_FACE);
            glEnable(GL_DEPTH_TEST);
            
            glEnable (GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(2.0f, 2.0f);
            glViewport(0, 0, cascade->resolution, cascade->resolution);
            
            shadowCascadeShader->bind();
            
            glClearColor(1.0f, 0.0f, 0.0f, 1.0f); // Debug red
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            // Fixme: should batch
            for (const auto & model : sceneObjects)
            {
                shadowCascadeShader->uniform("u_cascadeNear", (int) cascade->nearPlanes.size(), cascade->nearPlanes);
                shadowCascadeShader->uniform("u_cascadeFar", (int) cascade->farPlanes.size(), cascade->farPlanes);
                shadowCascadeShader->uniform("u_cascadeViewMatrixArray", (int) cascade->viewMatrices.size(), cascade->viewMatrices);
                shadowCascadeShader->uniform("u_cascadeProjMatrixArray", (int) cascade->projMatrices.size(), cascade->projMatrices);
                model.draw();
            }

            shadowCascadeShader->unbind();
            
            // Restore state
            shadowFbo.unbind();
            glViewport(0, 0, width, height);
            glDisable(GL_POLYGON_OFFSET_FILL);
            
            // Debug...
            //cascade->filter(float2(width, height));
            
        }
        
        {
            glClearColor(0.0f, 0.0f, 1.0f, 1.0f); // Debug blue
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
         
            // Draw scene
        }
        
        //skydome.render(viewProj, camera.get_eye_point(), camera.farClip);
        
        {
            ImGui::Text("Shadow Debug");
            ImGui::Separator();
            ImGui::Checkbox("Show Cascades", &showCascades);
        }

        //viewA->draw(uiSurface.children[0]->bounds, int2(width, height));
        //viewB->draw(uiSurface.children[1]->bounds, int2(width, height));
        
        gl_check_error(__FILE__, __LINE__);
        
        if (igm) igm->end_frame();
        
        glfwSwapBuffers(window);
    }
    
};
