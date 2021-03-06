#pragma once

#ifndef shader_monitor_h
#define shader_monitor_h

#include "GL_API.hpp"
#include "util.hpp"
#include "string_utils.hpp"

#include "third_party/efsw/efsw.hpp"

namespace avl
{

//@tofix - windows adds a trailing slash to paths returned by efsw!
class ShaderMonitor
{
    std::unique_ptr<efsw::FileWatcher> fileWatcher;
    
    struct ShaderAsset
    {
        GlShader & program;
        std::string vertexPath;
        std::string fragmentPath;
        bool shouldRecompile = false;
        ShaderAsset(GlShader & program, const std::string & v, const std::string & f) : program(program), vertexPath(v), fragmentPath(f) {};
    };
    
    struct UpdateListener : public efsw::FileWatchListener
    {
        std::function<void(const std::string filename)> callback;
        void handleFileAction(efsw::WatchID watchid, const std::string & dir, const std::string & filename, efsw::Action action, std::string oldFilename = "")
        {
            if (action == efsw::Actions::Modified)
            {
				std::cout << "Shader file updated: " << filename << std::endl;
                if (callback) callback(filename);
            }
        }
    };

    UpdateListener listener;
    
    std::vector<std::unique_ptr<ShaderAsset>> shaders;
    
public:
    
    ShaderMonitor()
    {
        fileWatcher.reset(new efsw::FileWatcher());
        
        efsw::WatchID id = fileWatcher->addWatch("assets/", &listener, true);
        
        listener.callback = [&](const std::string filename)
        {
            for (auto & shader : shaders)
            {
                if (filename == get_filename_with_extension(shader->vertexPath) || filename == get_filename_with_extension(shader->fragmentPath))
                {
                    shader->shouldRecompile = true;
                }
            }
        };
        fileWatcher->watch();
    }
    
    void add_shader(std::shared_ptr<GlShader> & program, const std::string & vertexShader, const std::string & fragmentShader)
    {
        shaders.emplace_back(new ShaderAsset(*program.get(), vertexShader, fragmentShader));
    }
    
    // Call this regularly on the gl thread
    void handle_recompile()
    {
        for (auto & shader : shaders)
        {
            if (shader->shouldRecompile)
            {
                try
                {
                    shader->program = GlShader(read_file_text(shader->vertexPath), read_file_text(shader->fragmentPath));
                    shader->shouldRecompile = false;
                }
                catch (const std::exception & e)
                {
                    std::cout << e.what() << std::endl;
                }
            }
        }
    }
    
};
    
}

#endif
