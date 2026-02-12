# Build Flappy C extension. Requires: Python dev headers, numpy, raylib.
# Install raylib: brew install raylib (macOS). Set RAYLIB_INC/RAYLIB_LIB if needed.
# Prefer project venv when present (from flappy dir: ../../../.venv = repo root .venv)
VENV_PYTHON := $(shell [ -f ../../.venv/bin/python ] && echo ../../.venv/bin/python)
PYTHON ?= $(or $(VENV_PYTHON),python3)
PYTHON_BASE := $(shell dirname $(PYTHON) 2>/dev/null || true)
ifeq ($(PYTHON_BASE),.)
  PYTHON := $(shell which python3 2>/dev/null || echo "python3")
endif
# Use the same Python's sysconfig so we link against the correct lib (venv often has no python3-config)
PYINC := $(shell $(PYTHON) -c "import sysconfig; i=sysconfig.get_path('include'); p=sysconfig.get_path('platinclude'); print('-I'+i+' -I'+p)" 2>/dev/null)
# Python lib: uv/cpython keeps dylib in prefix/lib
PYLIBDIR := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('prefix') + '/lib')" 2>/dev/null)
PYVER := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('VERSION'))" 2>/dev/null)
PYLDFLAGS := -L$(PYLIBDIR) -lpython$(PYVER)
NUMPY_INC := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())" 2>/dev/null)
ifeq (,$(NUMPY_INC))
  NUMPY_INC := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('purelib') + '/numpy/core/include')" 2>/dev/null)
endif
SO := binding$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))" 2>/dev/null)
ifeq (,$(SO))
  SO := binding.cpython-312-darwin.so
endif
RAYLIB_INC ?= -I/opt/homebrew/include -I/usr/local/include
RAYLIB_LIB ?= -L/opt/homebrew/lib -L/usr/local/lib -lraylib
CFLAGS := -O2 -fPIC $(PYINC) -I. $(RAYLIB_INC)
ifneq (,$(NUMPY_INC))
  CFLAGS += -I$(NUMPY_INC)
endif
LDFLAGS := $(PYLDFLAGS) $(RAYLIB_LIB) -shared

all: $(SO)

$(SO): binding.c flappy.h env_binding.h
	$(CC) $(CFLAGS) -o $@ binding.c $(LDFLAGS)

clean:
	rm -f $(SO) binding*.so

.PHONY: all clean
