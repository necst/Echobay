#!/usr/bin/env python
# encoding: utf-8

from waflib.Configure import conf

@conf
def check_yamlcpp(conf):
    # possible path to find headers
    includes_check = ['/usr/local/include/']
    libs_check = ['/usr/local/lib/']
    try:
      conf.start_msg('Checking for yaml-cpp includes\n')
      conf.find_file('yaml-cpp/yaml.h', includes_check)

      conf.find_file('libyaml-cpp.a', libs_check)

      conf.env.INCLUDES_YAMLCPP = includes_check
      conf.env.LIBPATH_YAMLCPP= libs_check
      conf.env.LIB_YAMLCPP = 'yaml-cpp'
      conf.env.DEFINES_YAMLCPP = ['USE_YAMLCPP']
      print(conf.env.INCLUDES_YAMLCPP)
      conf.end_msg('ok')

      
    except:
      conf.end_msg('Not found', 'RED')
      return
