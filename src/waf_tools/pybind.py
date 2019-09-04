#!/usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf

@conf
def check_pybind(conf):
    # possible path to find headers
    includes_check = ['/usr/local/include/', '/usr/include', '/usr/include/python3.6m']
    libs_check = ['/usr/local/lib/']
    try:
      conf.start_msg('Checking for pyBind includes\n')
      conf.find_file('pybind11/pybind11.h', includes_check)
      conf.find_file('python3.6m/Python.h', includes_check)

      #conf.find_file('libyaml-cpp.a', libs_check)

      conf.env.INCLUDES_PYBIND = includes_check
      conf.env.LIBPATH_PYBIND= libs_check
      conf.env.LIB_PYTHON = 'python3.6m'
      conf.env.DEFINES_PYBIND = ['USE_PYBIND']
      print(conf.env.INCLUDES_PYBIND)
      conf.end_msg('ok')

      
    except:
      conf.end_msg('Not found', 'RED')
      return
