#ifndef ui_space_h
#define ui_space_h

#include "geometric.hpp"

namespace avl
{
    
struct Space
{
    struct RenderEvent
    {
        Space * parent;
        void * user;
    };

    bool acceptInput = true;
    float aspectRatio = 1;
    URect placement = {{0,0},{0,0},{1,0},{1,0}};
    Bounds2D bounds;
    
    std::vector<std::shared_ptr<Space>> children;
    
    void add_child(const URect & placement, std::shared_ptr<Space> child = std::make_shared<Space>())
    {
        child->placement = placement;
        children.push_back(child);
    }
    
    void layout()
    {
        for (auto & child : children)
        {
            auto size = child->bounds.size();
            child->bounds = child->placement.resolve(bounds);
            auto childAspect = child->bounds.width() / child->bounds.height();
            if (childAspect > 0)
            {
                float xpadding = (1 - std::min((child->bounds.height() * childAspect) / child->bounds.width(), 1.0f)) / 2;
                float ypadding = (1 - std::min((child->bounds.width() / childAspect) / child->bounds.height(), 1.0f)) / 2;
                child->bounds = URect{{xpadding, 0}, {ypadding, 0}, {1 - xpadding, 0}, {1 - ypadding, 0}}.resolve(child->bounds);
            }
            if (child->bounds.size() != size) child->layout();
        }
    }
    
    virtual void render(const RenderEvent & e) {};
    virtual void input(const InputEvent & e) {};
    virtual void on_mouse_down(const float2 cursor) {};
    virtual void on_mouse_up(const float2 cursor) {};
    virtual void on_mouse_drag(const float2 cursor, const float2 delta) {};
};
    
}

#endif
