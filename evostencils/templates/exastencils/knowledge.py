from jinja2 import Template

template = Template("""
dimensionality                 = {{dimensionality}}

minLevel                       = {{min_level}}
maxLevel                       = {{max_level}}

discr_type                     = "{{discretization_type}}"
discr_defaultNeumannOrder      = {{default_neumann_order}}
discr_defaultDirichletOrder    = {{default_dirichlet_order}}
{%if grid_is_staggered%}
grid_isStaggered               = true
{%endif%}
{%if apply_shur_complement%}
experimental_applySchurCompl   = true
{%endif%}
// omp parallelization on exactly one fragment in one block
import '../lib/domain_onePatch.knowledge'
import '../lib/parallelization_pureOmp.knowledge'
""")

#print(template.render(dimensionality=2, min_level=0, max_level=8, discretization_type="FiniteDifferences", default_neumann_order=1, default_dirichlet_order=2,
#                      grid_is_staggered=True, apply_shur_complement=False))