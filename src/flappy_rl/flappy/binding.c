#include "flappy.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define Env Flappy
#include "env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->width = (int)unpack(kwargs, "width");
    env->height = (int)unpack(kwargs, "height");
    env->max_steps = 5000;
    PyObject* ms = PyDict_GetItemString(kwargs, "max_steps");
    if (ms != NULL && PyLong_Check(ms)) env->max_steps = (int)PyLong_AsLong(ms);
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
