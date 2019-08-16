from jinja2 import Template

template = Template("""
user                = "Guest"

configName          = "{{config_name}}"

basePathPrefix      = "./{{base_path}}/"

l3file              = "$configName$.exa3"
debugL1File         = "../Debug/$configName$_debug.exa1"
debugL2File         = "../Debug/$configName$_debug.exa2"
debugL3File         = "../Debug/$configName$_debug.exa3"
debugL4File         = "../Debug/$configName$_debug.exa4"

htmlLogFile         = "../Debug/$configName$_log.html"
outputPath          = "../generated/$configName$/"

produceHtmlLog      = true
timeStrategies      = true

buildfileGenerators = { "MakefileGenerator" }
""")