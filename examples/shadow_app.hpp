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

constexpr const char colorVertexShader[] = R"(#version 330
layout(location = 0) in vec3 vertex;
layout(location = 1) in vec3 vnorm;
uniform mat4 u_modelMatrix;
uniform mat4 u_modelMatrixIT;
uniform mat4 u_viewProj;
uniform vec3 u_color;
out vec3 color;
out vec3 normal;
void main()
{
    vec4 worldPos = u_modelMatrix * vec4(vertex, 1);
    gl_Position = u_viewProj * worldPos;
    color = u_color * 0.80;
    normal = normalize((u_modelMatrixIT * vec4(vnorm,0)).xyz);
}
)";

constexpr const char colorFragmentShader[] = R"(#version 330
in vec3 color;
out vec4 f_color;
in vec3 normal;
void main()
{
    f_color = (vec4(color.rgb, 1) * 0.75)+ (dot(normal, vec3(0, 1, 0)) * 0.33);
}
)";

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

// http://gamedev.stackexchange.com/questions/71058/how-extract-frustum-planes-from-clip-coordinates

inline std::array<float3, 4> make_near_clip_coords(GlCamera & cam, float aspectRatio)
{
    float3 viewDirection = normalize(cam.get_view_direction());
    float3 eye = cam.get_eye_point();
    
    auto m = cam.get_view_matrix();
    auto leftDir = m.getRow(0).xyz(); // Camera side direction
    auto upDir = m.getRow(1).xyz(); // Camera up direction
    
    auto coords = cam.make_frustum_coords(aspectRatio);
    
    float frustumTop	= coords[0];
    float frustumRight	= coords[1];
    float frustumBottom	= coords[2];
    float frustumLeft	= coords[3];
    
    float3 topLeft = eye + (cam.nearClip * viewDirection) + (frustumTop * upDir) + (frustumLeft * leftDir);
    float3 topRight = eye + (cam.nearClip  * viewDirection) + (frustumTop * upDir) + (frustumRight * leftDir);
    float3 bottomLeft = eye + (cam.nearClip  * viewDirection) + (frustumBottom * upDir) + (frustumLeft * leftDir);
    float3 bottomRight = eye + (cam.nearClip  * viewDirection) + (frustumBottom * upDir) + (frustumRight * leftDir);
    
    return {topLeft, topRight, bottomLeft, bottomRight};
}

inline std::array<float3, 4> make_far_clip_coords(GlCamera & cam, float aspectRatio)
{
    float3 viewDirection = normalize(cam.get_view_direction());
    float ratio = cam.farClip / cam.nearClip;
    float3 eye = cam.get_eye_point();
    
    auto m = cam.get_view_matrix();
    auto leftDir = m.getRow(0).xyz(); // Camera side direction
    auto upDir = m.getRow(1).xyz(); // Camera up direction
    
    auto coords = cam.make_frustum_coords(aspectRatio);
    
    float frustumTop	= coords[0];
    float frustumRight	= coords[1];
    float frustumBottom	= coords[2];
    float frustumLeft	= coords[3];
    
    float3 topLeft = eye + (cam.farClip * viewDirection) + (ratio * frustumTop * upDir) + (ratio * frustumLeft * leftDir);
    float3 topRight = eye + (cam.farClip * viewDirection) + (ratio * frustumTop * upDir) + (ratio * frustumRight * leftDir);
    float3 bottomLeft = eye + (cam.farClip * viewDirection) + (ratio * frustumBottom * upDir) + (ratio * frustumLeft * leftDir);
    float3 bottomRight = eye + (cam.farClip * viewDirection) + (ratio * frustumBottom * upDir) + (ratio * frustumRight * leftDir);
    
    return {topLeft, topRight, bottomLeft, bottomRight};
}

/*
struct Frustum
{
    Frustum()
    {
        for (int i = 0; i < 6; ++i) planes.push_back(Plane());
    }
    
    std::vector<Plane> planes;
};
*/

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
            // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch10.html
            float splitNear = i > 0 ? mix(near + (static_cast<float>(i) / 4.0f) * (far - near), near * pow(far / near, static_cast<float>(i) / 4.0f), splitLambda) : near;
            float splitFar = i < 4 - 1 ? mix(near + (static_cast<float>(i + 1) / 4.0f) * (far - near), near * pow(far / near, static_cast<float>(i + 1) / 4.0f), splitLambda) : far;

            GlCamera myCam = sceneCamera;
            myCam.nearClip = splitNear;
            myCam.farClip = splitFar;
            
            auto nc = make_near_clip_coords(myCam, aspectRatio);
            auto fc = make_far_clip_coords(myCam, aspectRatio);
            
            float3 splitVertices[8] = { nc[0], nc[1], nc[2], nc[3], fc[0], fc[1], fc[2], fc[3]};
            
            // Split centroid for the view matrix
            float3 splitCentroid = {0, 0, 0};
            for(size_t i = 0; i < 8; ++i)
            {
                splitCentroid += splitVertices[i];
            }
            splitCentroid /= 8.0f;
            
            //std::cout << splitCentroid << std::endl;
            
            // Make the view matrix
            float dist = max(splitFar - splitNear, distance(fc[0], fc[1]));
            //auto pose = look_at_pose(splitCentroid.xyz() - lightDir * dist, splitCentroid.xyz());
            //float4x4 viewMat = pose.matrix();
            
            //td::cout << pose.position << std::endl;
            myCam.look_at(splitCentroid - lightDir * dist, splitCentroid);
            auto viewMat = myCam.get_view_matrix();// make_view_matrix_from_pose(pose);
            
            //std::cout << viewMat << std::endl;

            // Xform split vertices to the light view space
            float3 splitVerticesLS[8];
            for (size_t i = 0; i < 8; ++i)
            {
                splitVerticesLS[i] = transform_coord(viewMat, splitVertices[i]);
            }
            
            // Find the frustum bounding box in viewspace
            float3 min = splitVerticesLS[0];
            float3 max = splitVerticesLS[0];
            for (size_t i = 1; i < 8; ++i)
            {
                min = avl::min(min, splitVerticesLS[i]);
                max = avl::max(max, splitVerticesLS[i]);
            }
            
            // Ortho projection matrix with the corners
            float nearOffset = 10.0f;
            float farOffset = 20.0f;
            float4x4 projMat = make_orthographic_matrix(min.x, max.x, min.y, max.y, -max.z - nearOffset, -min.z + farOffset);
            const float4x4 shadowBias = {{0.5f,0,0,0}, {0,0.5f,0,0}, {0,0,0.5f,0}, {0.5f,0.5f,0.5f,1}};
            
            viewMatrices.push_back(viewMat);
            projMatrices.push_back(projMat);
            shadowMatrices.push_back(shadowBias * projMat * viewMat);
            splitPlanes.push_back(float2(splitNear, splitFar));
            nearPlanes.push_back(-max.z - nearOffset);
            farPlanes.push_back(-min.z + farOffset);
            
            //std::cout << float2(splitNear, splitFar) << std::endl;
            //std::cout << offsetMat * projMat * viewMat << std::endl;
        }
        
    }
    
    void filter(float2 screen)
    {
        blurFramebuffer.bind_to_draw();
        
        glDisable(GL_ALPHA_TEST);
        glDisable(GL_BLEND);
        glViewport(0, 0, resolution, resolution);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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
        shadowArrayDepth.load_data(resolution, resolution, 4, GL_TEXTURE_2D_ARRAY, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, nullptr);
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
    std::random_device rd;
    std::mt19937 gen;
    
    GlCamera camera;
    PreethamProceduralSky skydome;
    FlyCameraController cameraController;
    ShaderMonitor shaderMonitor;
    Space uiSurface;
    
    std::unique_ptr<ShadowCascade> cascade;
    
    std::unique_ptr<gui::ImGuiManager> igm;
    
    std::shared_ptr<GLTextureView3D> viewA;
    std::shared_ptr<GLTextureView3D> viewB;
    std::shared_ptr<GLTextureView3D> viewC;
    std::shared_ptr<GLTextureView3D> viewD;
    
    std::vector<Renderable> sceneObjects;
    
    std::shared_ptr<GlShader> objectShader;
    std::shared_ptr<GlShader> gaussianBlurShader;
    std::shared_ptr<GlShader> shadowDebugShader;
    std::shared_ptr<GlShader> shadowCascadeShader;
    std::shared_ptr<GlShader> sceneCascadeShader;
    
    std::shared_ptr<GlShader> colorShader;
    
    Renderable floor;
    Renderable lightFrustum;
    
    float3 lightDir;
    bool showCascades = false;
    bool polygonOffset = false;
    
    ExperimentalApp() : GLFWApp(1280, 720, "Shadow Mapping App")
    {
        glfwSwapInterval(0);
        
        gen = std::mt19937(rd());
        
        igm.reset(new gui::ImGuiManager(window));
        gui::make_dark_theme();
        
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        glViewport(0, 0, width, height);
        
        cameraController.set_camera(&camera);
        camera.farClip = 75.f;
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
        
        colorShader.reset(new GlShader(colorVertexShader, colorFragmentShader));
        
        shadowDebugShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/debug_vert.glsl", "assets/shaders/shadow/debug_frag.glsl");
        
        //objectShader = make_watched_shader(shaderMonitor, "assets/shaders/simple_vert.glsl", "assets/shaders/simple_frag.glsl");
        
        gaussianBlurShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/gaussian_blur_vert.glsl", "assets/shaders/shadow/gaussian_blur_frag.glsl");
       
        shadowCascadeShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/shadowcascade_vert.glsl", "assets/shaders/shadow/shadowcascade_frag.glsl", "assets/shaders/shadow/shadowcascade_geom.glsl");
        sceneCascadeShader = make_watched_shader(shaderMonitor, "assets/shaders/shadow/cascade_vert.glsl", "assets/shaders/shadow/cascade_frag.glsl");
        
        cascade.reset(new ShadowCascade(shadowCascadeShader.get(), gaussianBlurShader.get()));
        
        viewA.reset(new GLTextureView3D(cascade->shadowArrayColor.get_gl_handle()));
        viewB.reset(new GLTextureView3D(cascade->shadowArrayColor.get_gl_handle()));
        viewC.reset(new GLTextureView3D(cascade->shadowArrayColor.get_gl_handle()));
        viewD.reset(new GLTextureView3D(cascade->shadowArrayColor.get_gl_handle()));
        
        auto hollowCube = load_geometry_from_ply("assets/models/geometry/CubeUniform.ply");
        for (auto & v : hollowCube.vertices) v *= 0.20f;
        //sceneObjects.push_back(Renderable(hollowCube));
        //sceneObjects.back().pose.position = float3(0, -5, 0);
        //sceneObjects.back().scale = float3(8, 8.f, 0.0001f);
        //sceneObjects.back().pose.orientation = make_rotation_quat_around_x(ANVIL_PI / 2);
        
        auto randomGeo = load_geometry_from_ply("assets/models/geometry/SphereUniform.ply");
        for (auto & v : randomGeo.vertices) v *= 0.0075f;
        
        auto r = std::uniform_real_distribution<float>(-32.0, 32.0);
        
        for (int i = 0; i < 64; ++i)
        {
            auto newObject = Renderable(randomGeo);
            newObject.pose.position = float3(r(gen), 0, r(gen));
            sceneObjects.push_back(std::move(newObject));
        }

        //sceneObjects.back().pose.position = float3(0, 0, 0);
        
        lightDir = normalize(float3( -1.4f, -0.37f, 0.63f ));
        
        floor = Renderable(make_plane(112.f, 112.f, 256, 256));
        lightFrustum = Renderable(make_frustum());
        lightFrustum.set_non_indexed(GL_LINES);
        
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
    
        const auto proj = camera.get_projection_matrix((float) width / (float) height);
        const float4x4 view = camera.get_view_matrix();
        const float4x4 viewProj = mul(proj, view);
        // model view proj is proj * view * model
        
        // Recreate cascades from camera view
        cascade->update(camera, lightDir, 1.f ); // aspect ratio?
        
        // Render shadowmaps

        {
            auto & shadowFbo = cascade->shadowArrayFramebuffer;
            shadowFbo.bind_to_draw();
            
            glEnable(GL_CULL_FACE);
            //glCullFace(GL_FRONT);
            
            glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);
            
            glDisable(GL_BLEND);
            
            //if (polygonOffset) glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(2.0f, 2.0f);
            
            glViewport(0, 0, cascade->resolution, cascade->resolution);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            shadowCascadeShader->bind();
            
            shadowCascadeShader->uniform("u_cascadeNear", (int) cascade->nearPlanes.size(), cascade->nearPlanes);
            shadowCascadeShader->uniform("u_cascadeFar", (int) cascade->farPlanes.size(), cascade->farPlanes);
            shadowCascadeShader->uniform("u_cascadeViewMatrixArray", (int) cascade->viewMatrices.size(), cascade->viewMatrices);
            shadowCascadeShader->uniform("u_cascadeProjMatrixArray", (int) cascade->projMatrices.size(), cascade->projMatrices);
            
            // Fixme: should batch
            for (const auto & model : sceneObjects)
            {
                model.draw();
            }

            shadowCascadeShader->unbind();
            
            // Restore state
            shadowFbo.unbind();
            glViewport(0, 0, width, height);
            //if (polygonOffset) glDisable(GL_POLYGON_OFFSET_FILL);
            
            //glCullFace(GL_BACK);
            gl_check_error(__FILE__, __LINE__);
        }
        
        // Fixme
        //cascade->filter(float2(width, height));
        
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        {
            // View space light dir
            float3 sceneLightDir = transform_vector(camera.get_view_matrix(), lightDir);
            
            sceneCascadeShader->bind();
            
            sceneCascadeShader->uniform("u_cascadesNear", (int) cascade->nearPlanes.size(), cascade->nearPlanes);
            sceneCascadeShader->uniform("u_cascadesFar", (int) cascade->farPlanes.size(), cascade->farPlanes);
            sceneCascadeShader->uniform("u_cascadesPlane", (int) cascade->splitPlanes.size(), cascade->splitPlanes);
            sceneCascadeShader->uniform("u_cascadesMatrix", (int) cascade->shadowMatrices.size(), cascade->shadowMatrices);
            sceneCascadeShader->texture("s_shadowMap", 0, GL_TEXTURE_2D_ARRAY, cascade->shadowArrayColor); // attachment 0
            sceneCascadeShader->uniform("u_lightDirection", sceneLightDir);
            sceneCascadeShader->uniform("u_expC", cascade->expCascade);
            sceneCascadeShader->uniform("u_showCascades", (float) showCascades);
            
            sceneCascadeShader->uniform("u_viewProjMatrix", viewProj);
            sceneCascadeShader->uniform("u_projMatrix", proj);
            
             // Todo - frustum intersection
            for (const auto & model : sceneObjects)
            {
                sceneCascadeShader->uniform("u_modelMatrix", model.get_model());
                sceneCascadeShader->uniform("u_modelMatrixIT", inv(transpose(model.get_model())));
                sceneCascadeShader->uniform("u_modelViewMatrix", view * model.get_model());
                sceneCascadeShader->uniform("u_normalMatrix",  get_rotation_submatrix(inv(transpose(view * model.get_model()))));
                model.draw();
            }
            
            {
                float4x4 model = make_translation_matrix({0, -10, 0}) * make_rotation_matrix({1, 0, 0}, -ANVIL_PI / 2) ;
                sceneCascadeShader->uniform("u_modelMatrix", model);
                sceneCascadeShader->uniform("u_modelMatrixIT", inv(transpose(model)));
                sceneCascadeShader->uniform("u_modelViewMatrix", view * model);
                sceneCascadeShader->uniform("u_normalMatrix", get_rotation_submatrix(inv(transpose(view * model))));
                floor.draw();
            }

            //glDisable(GL_DEPTH_TEST);
            //glDepthMask(GL_FALSE);
            
            sceneCascadeShader->unbind();
        }

        {
            colorShader->bind();
            
            auto pose = look_at_pose({0, 0, 0}, lightDir);
            auto model = make_view_matrix_from_pose(pose);
            colorShader->uniform("u_modelMatrix", model);
            colorShader->uniform("u_modelMatrixIT", inv(transpose(model)));
            colorShader->uniform("u_viewProj", viewProj);
            colorShader->uniform("u_color", float3(1, 0, 0));
            
            lightFrustum.draw();
            colorShader->unbind();
        }
        
        //skydome.render(viewProj, camera.get_eye_point(), camera.farClip);
        
        {
            ImGui::Checkbox("Show Cascades", &showCascades);
            ImGui::Checkbox("Use Polygon Offset", &polygonOffset);
            ImGui::SliderFloat("Shadow Factor", &cascade->expCascade, 0.f, 1000.f);
            ImGui::DragFloat3("Light Direction", &lightDir[0], 0.1f, -1.0f, 1.0f);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            // dnable/disable filtering
            // enable/disable mapping
            // float split lambda
            // camera near clip
            // camera far clip
            // Light direction
        }

        gl_check_error(__FILE__, __LINE__);
        
        viewA->draw(uiSurface.children[0]->bounds, int2(width, height), 0);
        viewB->draw(uiSurface.children[1]->bounds, int2(width, height), 1);
        viewC->draw(uiSurface.children[2]->bounds, int2(width, height), 2);
        viewD->draw(uiSurface.children[3]->bounds, int2(width, height), 3);
        
        //gl_check_error(__FILE__, __LINE__);
        
        if (igm) igm->end_frame();
        
        glfwSwapBuffers(window);
    }
    
};
