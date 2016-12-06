# Copyright (c) 2015-2016 Matthias Geier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Jupyter Notebook Tools for Sphinx.

http://nbsphinx.rtfd.org/

"""
__version__ = '0.2.11'

import copy
import json
import os
import re
import subprocess
try:
    from urllib.parse import unquote  # Python 3.x
except ImportError:
    from urllib2 import unquote  # Python 2.x

import docutils
from docutils.parsers import rst
import jinja2
import nbconvert
import nbformat
import sphinx
import sphinx.errors
import traitlets

_ipynbversion = 4

# See nbconvert/exporters/html.py:
DISPLAY_DATA_PRIORITY_HTML = (
    'application/javascript',
    'text/html',
    'text/markdown',
    'image/svg+xml',
    'text/latex',
    'image/png',
    'image/jpeg',
    'text/plain',
)
# See nbconvert/exporters/latex.py:
DISPLAY_DATA_PRIORITY_LATEX = (
    'text/latex',
    'application/pdf',
    'image/png',
    'image/jpeg',
    'image/svg+xml',
    'text/markdown',
    'text/plain',
)

RST_TEMPLATE = """
{% extends 'rst.tpl' %}


{% macro insert_empty_lines(text) %}
{%- set before, after = text | get_empty_lines %}
{%- if before %}
    :empty-lines-before: {{ before }}
{%- endif %}
{%- if after %}
    :empty-lines-after: {{ after }}
{%- endif %}
{%- endmacro %}


{% block any_cell %}
{%- if cell.metadata.nbsphinx != 'hidden' %}
{{ super() }}
{% endif %}
{%- endblock any_cell %}


{% block input -%}
.. nbinput:: {% if cell.metadata.magics_language -%}
{{ cell.metadata.magics_language }}
{%- elif nb.metadata.language_info -%}
{{ nb.metadata.language_info.pygments_lexer or nb.metadata.language_info.name }}
{%- else -%}
{{ resources.codecell_lexer }}
{%- endif -%}
{{ insert_empty_lines(cell.source) }}
{%- if cell.execution_count %}
    :execution-count: {{ cell.execution_count }}
{%- endif %}
{%- if not cell.outputs %}
    :no-output:
{%- endif %}
{%- if cell.source.strip() %}

{{ cell.source.strip('\n') | indent }}
{%- endif %}
{% endblock input %}


{% macro insert_nboutput(datatype, output, cell) -%}
.. nboutput::
{%- if datatype == 'text/plain' %}{# nothing #}
{%- else %} rst
{%- endif %}
{%- if output.output_type == 'execute_result' and cell.execution_count %}
    :execution-count: {{ cell.execution_count }}
{%- endif %}
{%- if output != cell.outputs[-1] %}
    :more-to-come:
{%- endif %}
{%- if output.name == 'stderr' %}
    :class: stderr
{%- endif %}
{%- if datatype == 'text/plain' -%}
{{ insert_empty_lines(output.data[datatype]) }}

{{ output.data[datatype].strip(\n) | indent }}
{%- elif datatype in ['image/svg+xml', 'image/png', 'image/jpeg', 'application/pdf'] %}

    .. image:: {{ output.metadata.filenames[datatype] | posix_path }}
{%- elif datatype in ['text/markdown'] %}

{{ output.data['text/markdown'] | markdown2rst | indent }}
{%- elif datatype in ['text/latex'] %}

    .. math::

{{ output.data['text/latex'] | strip_dollars | indent | indent }}
{%- elif datatype == 'text/html' %}

    .. raw:: html

{{ output.data['text/html'] | indent | indent }}
{%- elif datatype == 'application/javascript' %}
{% set div_id = uuid4() %}

    .. raw:: html

        <div id="{{ div_id }}"></div>
        <script type="text/javascript">
        var element = document.getElementById('{{ div_id }}');
{{ output.data['application/javascript'] | indent | indent }}
        </script>
{%- elif datatype == 'ansi' %}

    .. rst-class:: highlight

    .. raw:: html

        <pre>
{{ output.data[datatype] | ansi2html | indent | indent }}
        </pre>

    .. raw:: latex

        % This comment is needed to force a line break for adjacent ANSI cells
        \\begin{OriginalVerbatim}[commandchars=\\\\\\{\\}]
{{ output.data[datatype] | ansi2latex | indent | indent }}
        \\end{OriginalVerbatim}
{% else %}

    .. nbwarning:: Data type cannot be displayed: {{ datatype }}
{%- endif %}
{% endmacro %}


{% block nboutput -%}
{%- set html_datatype, latex_datatype = output | get_output_type %}
{%- if html_datatype == latex_datatype %}
{{ insert_nboutput(html_datatype, output, cell) }}
{%- else %}
.. only:: html

{{ insert_nboutput(html_datatype, output, cell) | indent }}
.. only:: latex

{{ insert_nboutput(latex_datatype, output, cell) | indent }}
{%- endif %}
{% endblock nboutput %}


{% block execute_result %}{{ self.nboutput() }}{% endblock execute_result %}
{% block display_data %}{{ self.nboutput() }}{% endblock display_data %}
{% block stream %}{{ self.nboutput() }}{% endblock stream %}
{% block error %}{{ self.nboutput() }}{% endblock error %}


{% block markdowncell %}
{%- if 'nbsphinx-toctree' in cell.metadata %}
{{ cell | extract_toctree }}
{%- else %}
{{ super() }}
{% endif %}
{% endblock markdowncell %}


{% block rawcell %}
{%- set raw_mimetype = cell.metadata.get('raw_mimetype', '').lower() %}
{%- if raw_mimetype == '' %}
.. raw:: html

{{ cell.source | indent }}

.. raw:: latex

{{ cell.source | indent }}
{%- elif raw_mimetype == 'text/html' %}
.. raw:: html

{{ cell.source | indent }}
{%- elif raw_mimetype == 'text/latex' %}
.. raw:: latex

{{ cell.source | indent }}
{%- elif raw_mimetype == 'text/markdown' %}
{{ cell.source | markdown2rst }}
{%- elif raw_mimetype == 'text/restructuredtext' %}
{{ cell.source }}
{% endif %}
{% endblock rawcell %}
"""


LATEX_PREAMBLE = r"""
% Jupyter Notebook prompt colors
\definecolor{nbsphinxin}{HTML}{303F9F}
\definecolor{nbsphinxout}{HTML}{D84315}
% ANSI colors for output streams and traceback highlighting
\definecolor{ansi-black}{HTML}{3E424D}
\definecolor{ansi-black-intense}{HTML}{282C36}
\definecolor{ansi-red}{HTML}{E75C58}
\definecolor{ansi-red-intense}{HTML}{B22B31}
\definecolor{ansi-green}{HTML}{00A250}
\definecolor{ansi-green-intense}{HTML}{007427}
\definecolor{ansi-yellow}{HTML}{DDB62B}
\definecolor{ansi-yellow-intense}{HTML}{B27D12}
\definecolor{ansi-blue}{HTML}{208FFB}
\definecolor{ansi-blue-intense}{HTML}{0065CA}
\definecolor{ansi-magenta}{HTML}{D160C4}
\definecolor{ansi-magenta-intense}{HTML}{A03196}
\definecolor{ansi-cyan}{HTML}{60C6C8}
\definecolor{ansi-cyan-intense}{HTML}{258F8F}
\definecolor{ansi-white}{HTML}{C5C1B4}
\definecolor{ansi-white-intense}{HTML}{A1A6B2}
"""


CSS_STRING = """
/* CSS for nbsphinx extension */

/* remove conflicting styling from Sphinx themes */
div.nbinput,
div.nbinput > div,
div.nbinput div[class^=highlight],
div.nbinput div[class^=highlight] pre,
div.nboutput,
div.nboutput > div,
div.nboutput div[class^=highlight],
div.nboutput div[class^=highlight] pre {
    background: none;
    border: none;
    padding: 0 0;
    margin: 0;
    box-shadow: none;
}

/* avoid gaps between output lines */
div.nboutput div[class^=highlight] pre {
    line-height: normal;
}

/* input/output containers */
div.nbinput,
div.nboutput {
    display: -webkit-flex;
    display: flex;
    align-items: flex-start;
    margin: 0;
}

/* input container */
div.nbinput {
    padding-top: 5px;
}

/* last container */
div.nblast {
    padding-bottom: 5px;
}

/* input prompt */
div.nbinput > :first-child pre {
    color: #303F9F;
}

/* output prompt */
div.nboutput > :first-child pre {
    color: #D84315;
}

/* all prompts */
div.nbinput > :first-child[class^=highlight],
div.nboutput > :first-child[class^=highlight],
div.nboutput > :first-child {
    min-width: %(nbsphinx_prompt_width)s;
    padding-top: 0.4em;
    padding-right: 0.4em;
    text-align: right;
    flex: 0;
}

/* input/output area */
div.nbinput > :nth-child(2)[class^=highlight],
div.nboutput > :nth-child(2),
div.nboutput > :nth-child(2)[class^=highlight] {
    padding: 0.4em;
    -webkit-flex: 1;
    flex: 1;
    overflow: auto;
}

/* input area */
div.nbinput > :nth-child(2)[class^=highlight] {
    border: 1px solid #cfcfcf;
    border-radius: 2px;
    background: #f7f7f7;
}

/* override MathJax center alignment in output cells */
div.nboutput div[class*=MathJax] {
    text-align: left !important;
}

/* override sphinx.ext.pngmath center alignment in output cells */
div.nboutput div.math p {
    text-align: left;
}

/* standard error */
div.nboutput  > :nth-child(2).stderr {
    background: #fdd;
}

/* ANSI colors */
.ansi-black-fg { color: #3E424D; }
.ansi-black-bg { background-color: #3E424D; }
.ansi-black-intense-fg { color: #282C36; }
.ansi-black-intense-bg { background-color: #282C36; }
.ansi-red-fg { color: #E75C58; }
.ansi-red-bg { background-color: #E75C58; }
.ansi-red-intense-fg { color: #B22B31; }
.ansi-red-intense-bg { background-color: #B22B31; }
.ansi-green-fg { color: #00A250; }
.ansi-green-bg { background-color: #00A250; }
.ansi-green-intense-fg { color: #007427; }
.ansi-green-intense-bg { background-color: #007427; }
.ansi-yellow-fg { color: #DDB62B; }
.ansi-yellow-bg { background-color: #DDB62B; }
.ansi-yellow-intense-fg { color: #B27D12; }
.ansi-yellow-intense-bg { background-color: #B27D12; }
.ansi-blue-fg { color: #208FFB; }
.ansi-blue-bg { background-color: #208FFB; }
.ansi-blue-intense-fg { color: #0065CA; }
.ansi-blue-intense-bg { background-color: #0065CA; }
.ansi-magenta-fg { color: #D160C4; }
.ansi-magenta-bg { background-color: #D160C4; }
.ansi-magenta-intense-fg { color: #A03196; }
.ansi-magenta-intense-bg { background-color: #A03196; }
.ansi-cyan-fg { color: #60C6C8; }
.ansi-cyan-bg { background-color: #60C6C8; }
.ansi-cyan-intense-fg { color: #258F8F; }
.ansi-cyan-intense-bg { background-color: #258F8F; }
.ansi-white-fg { color: #C5C1B4; }
.ansi-white-bg { background-color: #C5C1B4; }
.ansi-white-intense-fg { color: #A1A6B2; }
.ansi-white-intense-bg { background-color: #A1A6B2; }

.ansi-bold { font-weight: bold; }
"""

CSS_STRING_READTHEDOCS = """
/* CSS overrides for sphinx_rtd_theme */

/* 24px margin */
.nbinput.nblast,
.nboutput.nblast {
    margin-bottom: 19px;  /* padding has already 5px */
}

/* ... except between code cells! */
.nblast + .nbinput {
    margin-top: -19px;
}

/* nice headers on first paragraph of info/warning boxes */
.admonition .first {
    margin: -12px;
    padding: 6px 12px;
    margin-bottom: 12px;
    color: #fff;
    line-height: 1;
    display: block;
}
.admonition.warning .first {
    background: #f0b37e;
}
.admonition.note .first {
    background: #6ab0de;
}
.admonition > p:before {
    margin-right: 4px;  /* make room for the exclamation icon */
}
"""


class Exporter(nbconvert.RSTExporter):
    """Convert Jupyter notebooks to reStructuredText.

    Uses nbconvert to convert Jupyter notebooks to a reStructuredText
    string with custom reST directives for input and output cells.

    Notebooks without output cells are automatically executed before
    conversion.

    """

    def __init__(self, execute='auto', execute_arguments=[],
                 allow_errors=False, timeout=30, codecell_lexer='none'):
        """Initialize the Exporter."""
        self._execute = execute
        self._execute_arguments = execute_arguments
        self._allow_errors = allow_errors
        self._timeout = timeout
        self._codecell_lexer = codecell_lexer
        loader = jinja2.DictLoader({'nbsphinx-rst.tpl': RST_TEMPLATE})
        super(Exporter, self).__init__(
            template_file='nbsphinx-rst', extra_loaders=[loader],
            filters={
                'markdown2rst': markdown2rst,
                'get_empty_lines': _get_empty_lines,
                'extract_toctree': _extract_toctree,
                'get_output_type': _get_output_type,
            })

    @property
    def default_config(self):
        c = traitlets.config.Config(
            {'HighlightMagicsPreprocessor': {'enabled': True}})
        c.merge(super(Exporter, self).default_config)
        return c

    def from_notebook_node(self, nb, resources=None, **kw):
        nb = copy.deepcopy(nb)
        if resources is None:
            resources = {}
        else:
            resources = copy.deepcopy(resources)
        # Set default codecell lexer
        resources['codecell_lexer'] = self._codecell_lexer

        nbsphinx_metadata = nb.metadata.get('nbsphinx', {})

        execute = nbsphinx_metadata.get('execute', self._execute)
        if execute not in ('always', 'never', 'auto'):
            raise ValueError('invalid execute option: {!r}'.format(execute))
        auto_execute = (
            # At least one code cell actually containing source code:
            any(c.source for c in nb.cells if c.cell_type == 'code') and
            # No outputs, not even a prompt number:
            not any(c.get('outputs') or c.get('execution_count')
                    for c in nb.cells if c.cell_type == 'code')
        )
        if execute == 'always' or (execute == 'auto' and auto_execute):
            allow_errors = nbsphinx_metadata.get(
                'allow_errors', self._allow_errors)
            timeout = nbsphinx_metadata.get('timeout', self._timeout)
            pp = nbconvert.preprocessors.ExecutePreprocessor(
                extra_arguments=self._execute_arguments,
                allow_errors=allow_errors, timeout=timeout)
            nb, resources = pp.preprocess(nb, resources)

        # Call into RSTExporter
        rststr, resources = super(Exporter, self).from_notebook_node(
            nb, resources, **kw)

        orphan = nbsphinx_metadata.get('orphan', False)
        if orphan is True:
            rststr = ':orphan:\n\n' + rststr
        elif orphan is not False:
            raise ValueError('invalid orphan option: {!r}'.format(orphan))

        return rststr, resources


class NotebookParser(rst.Parser):
    """Sphinx source parser for Jupyter notebooks.

    Uses nbsphinx.Exporter to convert notebook content to a
    reStructuredText string, which is then parsed by Sphinx's built-in
    reST parser.

    """

    def get_transforms(self):
        """List of transforms for documents parsed by this parser."""
        return rst.Parser.get_transforms(self) + [
            ProcessLocalLinks, CreateSectionLabels, ReplaceAlertDivs]

    def parse(self, inputstring, document):
        """Parse `inputstring`, write results to `document`."""
        nb = nbformat.reads(inputstring, as_version=_ipynbversion)
        env = document.settings.env
        srcdir = os.path.dirname(env.doc2path(env.docname))
        auxdir = os.path.join(env.doctreedir, 'nbsphinx')
        sphinx.util.ensuredir(auxdir)

        resources = {}
        # Working directory for ExecutePreprocessor
        resources['metadata'] = {'path': srcdir}
        # Sphinx doesn't accept absolute paths in images etc.
        resources['output_files_dir'] = os.path.relpath(auxdir, srcdir)
        resources['unique_key'] = re.sub('[/ ]', '_', env.docname)

        exporter = Exporter(
            execute=env.config.nbsphinx_execute,
            execute_arguments=env.config.nbsphinx_execute_arguments,
            allow_errors=env.config.nbsphinx_allow_errors,
            timeout=env.config.nbsphinx_timeout,
            codecell_lexer=env.config.nbsphinx_codecell_lexer,
        )

        try:
            rststring, resources = exporter.from_notebook_node(nb, resources)
        except nbconvert.preprocessors.execute.CellExecutionError as e:
            lines = str(e).split('\n')
            lines[0] = 'CellExecutionError in {}:'.format(
                env.doc2path(env.docname, base=None))
            lines.append("You can ignore this error by setting the following "
                         "in conf.py:\n\n    nbsphinx_allow_errors = True\n")
            raise NotebookError('\n'.join(lines))
        except Exception as e:
            raise NotebookError(type(e).__name__ + ' in ' +
                                env.doc2path(env.docname, base=None) + ':\n' +
                                str(e))

        # Create additional output files (figures etc.),
        # see nbconvert.writers.FilesWriter.write()
        for filename, data in resources.get('outputs', {}).items():
            dest = os.path.normpath(os.path.join(srcdir, filename))
            with open(dest, 'wb') as f:
                f.write(data)

        rst.Parser.parse(self, rststring, document)


class NotebookError(sphinx.errors.SphinxError):
    """Error during notebook parsing."""

    category = 'Notebook error'


class CodeNode(docutils.nodes.Element):
    """A custom node that contains a literal_block node."""

    @classmethod
    def create(cls, text, language='none'):
        """Create a new CodeNode containing a literal_block node.

        Apparently, this cannot be done in CodeNode.__init__(), see:
        https://groups.google.com/forum/#!topic/sphinx-dev/0chv7BsYuW0

        """
        node = docutils.nodes.literal_block(text, text, language=language)
        return cls(text, node)


class AdmonitionNode(docutils.nodes.Element):
    """A custom node for info and warning boxes."""


# See http://docutils.sourceforge.net/docs/howto/rst-directives.html

class NbInput(rst.Directive):
    """A notebook input cell with prompt and code area."""

    required_arguments = 0
    optional_arguments = 1  # lexer name
    final_argument_whitespace = False
    option_spec = {
        'execution-count': rst.directives.positive_int,
        'empty-lines-before': rst.directives.nonnegative_int,
        'empty-lines-after': rst.directives.nonnegative_int,
        'no-output': rst.directives.flag,
    }
    has_content = True

    def run(self):
        """This is called by the reST parser."""
        execution_count = self.options.get('execution-count')
        classes = ['nbinput']
        if 'no-output' in self.options:
            classes.append('nblast')
        container = docutils.nodes.container(classes=classes)

        # Input prompt
        text = 'In [{}]:'.format(execution_count if execution_count else ' ')
        container += CodeNode.create(text)
        latex_prompt = text + ' '

        # Input code area
        text = '\n'.join(self.content.data)
        node = CodeNode.create(
            text, language=self.arguments[0] if self.arguments else 'none')
        _set_empty_lines(node, self.options)
        node.attributes['latex_prompt'] = latex_prompt
        container += node
        self.state.document['nbsphinx_include_css'] = True
        return [container]


class NbOutput(rst.Directive):
    """A notebook output cell with optional prompt."""

    required_arguments = 0
    optional_arguments = 1  # 'rst' or nothing (which means literal text)
    final_argument_whitespace = False
    option_spec = {
        'execution-count': rst.directives.positive_int,
        'more-to-come': rst.directives.flag,
        'empty-lines-before': rst.directives.nonnegative_int,
        'empty-lines-after': rst.directives.nonnegative_int,
        'class': rst.directives.unchanged,
    }
    has_content = True

    def run(self):
        """This is called by the reST parser."""
        outputtype = self.arguments[0] if self.arguments else ''
        execution_count = self.options.get('execution-count')
        classes = ['nboutput']
        if 'more-to-come' not in self.options:
            classes.append('nblast')
        container = docutils.nodes.container(classes=classes)

        # Optional output prompt
        if execution_count:
            text = 'Out[{}]:'.format(execution_count)
            container += CodeNode.create(text)
            latex_prompt = text + ' '
        else:
            container += rst.nodes.container()  # empty container for HTML
            latex_prompt = ''

        # Output area
        if outputtype == 'rst':
            classes = [self.options.get('class', '')]
            output_area = docutils.nodes.container(classes=classes)
            self.state.nested_parse(self.content, self.content_offset,
                                    output_area)
            container += output_area
        else:
            text = '\n'.join(self.content.data)
            node = CodeNode.create(text)
            _set_empty_lines(node, self.options)
            node.attributes['latex_prompt'] = latex_prompt
            container += node
        self.state.document['nbsphinx_include_css'] = True
        return [container]


class _NbAdmonition(rst.Directive):
    """Base class for NbInfo and NbWarning."""

    required_arguments = 0
    optional_arguments = 0
    option_spec = {}
    has_content = True

    def run(self):
        """This is called by the reST parser."""
        node = AdmonitionNode(classes=['admonition', self._class])
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class NbInfo(_NbAdmonition):
    """An information box."""

    _class = 'note'


class NbWarning(_NbAdmonition):
    """A warning box."""

    _class = 'warning'


def markdown2rst(text):
    """Convert a Markdown string to reST via pandoc.

    This is very similar to nbconvert.filters.markdown.markdown2rst(),
    except that it uses a pandoc filter to convert raw LaTeX blocks to
    "math" directives (instead of "raw:: latex" directives).

    """

    def rawlatex2math_hook(obj):
        if obj.get('t') == 'RawBlock' and obj['c'][0] == 'latex':
            obj['t'] = 'Para'
            obj['c'] = [{
                't': 'Math',
                'c': [
                    {'t': 'DisplayMath', 'c': []},
                    obj['c'][1],
                ]
            }]
        return obj

    def rawlatex2math(text):
        json_data = json.loads(text, object_hook=rawlatex2math_hook)
        return json.dumps(json_data)

    rststring = pandoc(text, 'markdown', 'rst', filter_func=rawlatex2math)
    return re.sub(r'^(\s*)\.\. math::$',
                  r'\1.. math::\1   :nowrap:',
                  rststring,
                  flags=re.MULTILINE)


def pandoc(source, fmt, to, filter_func=None):
    """Convert a string in format `from` to format `to` via pandoc.

    This is based on nbconvert.utils.pandoc.pandoc() and extended to
    allow passing a filter function.

    """
    def encode(text):
        return text if isinstance(text, bytes) else text.encode('utf-8')

    def decode(data):
        return data.decode('utf-8') if isinstance(data, bytes) else data

    cmd1 = ['pandoc', '--from', fmt, '--to', 'json']
    cmd2 = ['pandoc', '--from', 'json', '--to', to]

    nbconvert.utils.pandoc.check_pandoc_version()

    p = subprocess.Popen(cmd1, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    json_data, _ = p.communicate(encode(source))

    if filter_func:
        json_data = encode(filter_func(decode(json_data)))

    p = subprocess.Popen(cmd2, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, _ = p.communicate(json_data)
    return decode(out).rstrip('\n')


def _extract_toctree(cell):
    """Extract links from Markdown cell and create toctree."""
    lines = ['.. toctree::']
    options = cell.metadata['nbsphinx-toctree']
    try:
        for option, value in options.items():
            if value is True:
                lines.append(':{}:'.format(option))
            elif value is False:
                pass
            else:
                lines.append(':{}: {}'.format(option, value))
    except AttributeError:
        raise ValueError(
            'invalid nbsphinx-toctree option: {!r}'.format(options))

    text = nbconvert.filters.markdown2rst(cell.source)
    settings = docutils.frontend.OptionParser(
        components=(rst.Parser,)).get_default_values()
    toctree_node = docutils.utils.new_document('extract_toctree', settings)
    parser = rst.Parser()
    parser.parse(text, toctree_node)

    if 'caption' not in options:
        for sec in toctree_node.traverse(docutils.nodes.section):
            assert sec.children
            assert isinstance(sec.children[0], docutils.nodes.title)
            title = sec.children[0].astext()
            lines.append(':caption: ' + title)
            break
    lines.append('')  # empty line
    for ref in toctree_node.traverse(docutils.nodes.reference):
        lines.append(ref.astext().replace('\n', '') +
                     ' <' + unquote(ref.get('refuri')) + '>')
    return '\n    '.join(lines)


def _get_empty_lines(text):
    """Get number of empty lines before and after code."""
    before = len(text) - len(text.lstrip('\n'))
    after = len(text) - len(text.strip('\n')) - before
    return before, after


def _get_output_type(output):
    """Choose appropriate output data types for HTML and LaTeX."""
    if output.output_type == 'stream':
        html_datatype = latex_datatype = 'ansi'
        text = output.text
        output.data = {'ansi': text[:-1] if text.endswith('\n') else text}
    elif output.output_type == 'error':
        html_datatype = latex_datatype = 'ansi'
        output.data = {'ansi': '\n'.join(output.traceback)}
    else:
        for datatype in DISPLAY_DATA_PRIORITY_HTML:
            if datatype in output.data:
                html_datatype = datatype
                break
        else:
            html_datatype = ', '.join(output.data.keys())
        for datatype in DISPLAY_DATA_PRIORITY_LATEX:
            if datatype in output.data:
                latex_datatype = datatype
                break
        else:
            latex_datatype = ', '.join(output.data.keys())
    return html_datatype, latex_datatype


def _set_empty_lines(node, options):
    """Set "empty lines" attributes on a CodeNode.

    See http://stackoverflow.com/q/34050044/500098.

    """
    for attr in 'empty-lines-before', 'empty-lines-after':
        value = options.get(attr, 0)
        if value:
            node.attributes[attr] = value


class ProcessLocalLinks(docutils.transforms.Transform):
    """Process links to local files.

    Marks local files to be copied to the HTML output directory and
    turns links to local notebooks into ``:doc:``/``:ref:`` links.

    Links to subsections are possible with ``...#Subsection-Title``.
    These links use the labels created by CreateSectionLabels.

    Links to subsections use ``:ref:``, links to whole notebooks use
    ``:doc:``.  Latter can be useful if you have an ``index.rst`` but
    also want a distinct ``index.ipynb`` for use with Jupyter.
    In this case you can use such a link in a notebook::

        [Back to main page](index.ipynb)

    In Jupyter, this will create a "normal" link to ``index.ipynb``, but
    in the files generated by Sphinx, this will become a link to the
    main page created from ``index.rst``.

    """

    default_priority = 400  # Should probably be adjusted?

    def apply(self):
        env = self.document.settings.env
        for node in self.document.traverse(docutils.nodes.reference):
            uri = node.get('refuri', '')
            if not uri:
                continue  # No URI (e.g. named reference)
            elif '://' in uri:
                continue  # Not a local link
            elif uri.startswith('#') or uri.startswith('mailto:'):
                continue  # Nothing to be done

            unquoted_uri = unquote(uri)
            for suffix in env.config.source_suffix:
                if unquoted_uri.lower().endswith(suffix.lower()):
                    target = unquoted_uri[:-len(suffix)]
                    break
            else:
                target = ''

            if target:
                target_ext = ''
                reftype = 'doc'
                refdomain = None
            elif '.ipynb#' in uri.lower():
                idx = uri.lower().find('.ipynb#')
                target = unquote(uri[:idx])
                target_ext = uri[idx:]
                reftype = 'ref'
                refdomain = 'std'
            else:
                file = os.path.normpath(
                    os.path.join(os.path.dirname(env.docname), unquoted_uri))
                if not os.path.isfile(os.path.join(env.srcdir, file)):
                    env.app.warn('file not found: {!r}'.format(file),
                                 env.doc2path(env.docname))
                    continue  # Link is ignored
                elif file.startswith('..'):
                    env.app.warn(
                        'link outside of source directory: {!r}'.format(file),
                        env.doc2path(env.docname))
                    continue  # Link is ignored
                if not hasattr(env, 'nbsphinx_files'):
                    env.nbsphinx_files = {}
                env.nbsphinx_files.setdefault(env.docname, []).append(file)
                continue  # We're done here

            target_docname = os.path.normpath(
                os.path.join(os.path.dirname(env.docname), target))
            if target_docname in env.found_docs:
                if target_ext:
                    target = target_docname + target_ext
                    target = target.lower()
                linktext = node.astext()
                xref = sphinx.addnodes.pending_xref(
                    reftype=reftype, reftarget=target, refdomain=refdomain,
                    refwarn=True, refexplicit=True, refdoc=env.docname)
                xref += docutils.nodes.Text(linktext, linktext)
                node.replace_self(xref)


class CreateSectionLabels(docutils.transforms.Transform):
    """Make labels for each notebook and each section thereof.

    These labels are referenced in ProcessLocalLinks.
    Note: Sphinx lower-cases the HTML section IDs, Jupyter doesn't.

    """

    default_priority = 250  # Before references.PropagateTargets (260)

    def apply(self):
        env = self.document.settings.env
        i_still_have_to_create_the_notebook_label = True
        for section in self.document.traverse(docutils.nodes.section):
            assert section.children
            assert isinstance(section.children[0], docutils.nodes.title)
            title = section.children[0].astext()
            link_id = title.replace(' ', '-')
            section['ids'] = [link_id]
            label = env.docname + '.ipynb#' + link_id
            label = label.lower()
            env.domaindata['std']['labels'][label] = (
                env.docname, link_id, title)
            env.domaindata['std']['anonlabels'][label] = (
                env.docname, link_id)

            # Create a label for the whole notebook using the first section:
            if i_still_have_to_create_the_notebook_label:
                label = env.docname.lower() + '.ipynb'
                env.domaindata['std']['labels'][label] = (
                    env.docname, '', title)
                env.domaindata['std']['anonlabels'][label] = (
                    env.docname, '')
                i_still_have_to_create_the_notebook_label = False


class ReplaceAlertDivs(docutils.transforms.Transform):
    """Replace certain <div> elements with AdmonitionNode containers.

    This is a quick-and-dirty work-around until a proper
    Mardown/CommonMark extension for note/warning boxes is available.

    """

    default_priority = 500  # Doesn't really matter

    _start_re = re.compile(
        r'\s*<div\s*class\s*=\s*(?P<q>"|\')([a-z\s-]*)(?P=q)\s*>\s*\Z',
        flags=re.IGNORECASE)
    _class_re = re.compile(r'\s*alert\s*alert-(info|warning)\s*\Z')
    _end_re = re.compile(r'\s*</div\s*>\s*\Z', flags=re.IGNORECASE)

    def apply(self):
        start_tags = []
        for node in self.document.traverse(docutils.nodes.raw):
            if node['format'] != 'html':
                continue
            start_match = self._start_re.match(node.astext())
            if not start_match:
                continue
            class_match = self._class_re.match(start_match.group(2))
            if not class_match:
                continue
            admonition_class = class_match.group(1)
            if admonition_class == 'info':
                admonition_class = 'note'
            start_tags.append((node, admonition_class))

        # Reversed order to allow nested <div> elements:
        for node, admonition_class in reversed(start_tags):
            content = []
            for sibling in node.traverse(include_self=False, descend=False,
                                         siblings=True, ascend=False):
                end_tag = (isinstance(sibling, docutils.nodes.raw) and
                           sibling['format'] == 'html' and
                           self._end_re.match(sibling.astext()))
                if end_tag:
                    admonition_node = AdmonitionNode(
                        classes=['admonition', admonition_class])
                    admonition_node.extend(content)
                    parent = node.parent
                    parent.replace(node, admonition_node)
                    for n in content:
                        parent.remove(n)
                    parent.remove(sibling)
                    break
                else:
                    content.append(sibling)


def builder_inited(app):
    # Add color definitions to LaTeX preamble
    latex_elements = app.builder.config.latex_elements
    latex_elements['preamble'] = '\n'.join([
        LATEX_PREAMBLE,
        latex_elements.get('preamble', ''),
    ])

    # Set default value for CSS prompt width
    if app.config.nbsphinx_prompt_width is None:
        app.config.nbsphinx_prompt_width = {
            'agogo': '7ex',
            'better': '8ex',
            'classic': '7ex',
            'cloud': '8ex',
            'dotted': '8ex',
            'haiku': '7ex',
            'julia': '7ex',
            'nature': '8ex',
            'pyramid': '8ex',
            'redcloud': '8ex',
            'sphinx_py3doc_enhanced_theme': '8ex',
            'sphinx_rtd_theme': '8ex',
            'traditional': '6ex',
        }.get(app.config.html_theme, '9ex')


def html_page_context(app, pagename, templatename, context, doctree):
    """Add CSS string to HTML pages that contain code cells."""
    style = ''
    if doctree and doctree.get('nbsphinx_include_css'):
        style += CSS_STRING % app.config
    if doctree and app.config.html_theme in ('sphinx_rtd_theme', 'julia'):
        style += CSS_STRING_READTHEDOCS
    if style:
        context['body'] = '\n<style>' + style + '</style>\n' + context['body']


def html_collect_pages(app):
    """This event handler is abused to copy local files around."""
    files = set()
    for file_list in getattr(app.env, 'nbsphinx_files', {}).values():
        files.update(file_list)
    for file in app.status_iterator(files, 'copying linked files... ',
                                    sphinx.util.console.brown, len(files)):
        target = os.path.join(app.builder.outdir, file)
        sphinx.util.ensuredir(os.path.dirname(target))
        try:
            sphinx.util.copyfile(os.path.join(app.env.srcdir, file), target)
        except OSError as err:
            app.warn('cannot copy local file {!r}: {}'.format(file, err))
    return []  # No new HTML pages are created


def env_purge_doc(app, env, docname):
    """Remove list of local files for a given document."""
    try:
        del env.nbsphinx_files[docname]
    except (AttributeError, KeyError):
        pass


def depart_code_html(self, node):
    """Add empty lines before and after the code."""
    text = self.body[-1]
    text = text.replace('<pre>',
                        '<pre>\n' + '\n' * node.get('empty-lines-before', 0))
    text = text.replace('</pre>',
                        '\n' * node.get('empty-lines-after', 0) + '</pre>')
    self.body[-1] = text


def visit_code_latex(self, node):
    """Avoid creating a separate prompt node.

    The prompt will be pre-pended in the main code node.

    """
    if 'latex_prompt' not in node.attributes:
        raise docutils.nodes.SkipNode()


def depart_code_latex(self, node):
    """Some changes to code blocks.

    * Remove the frame (by changing Verbatim -> OriginalVerbatim)
    * Add empty lines before and after the code
    * Add prompt to the first line, empty space to the following lines

    """
    lines = self.body[-1].split('\n')
    out = []
    assert lines[0] == ''
    out.append(lines[0])
    if lines[1].startswith(r'\begin{sphinxVerbatim}'):  # Sphinx >= 1.5
        out.append(lines[1].replace('sphinxVerbatim', 'Verbatim'))
    elif lines[1].startswith(r'\begin{Verbatim}'):  # Sphinx < 1.5
        out.append(lines[1].replace('Verbatim', 'OriginalVerbatim'))
    else:
        assert False
    code_lines = (
        [''] * node.get('empty-lines-before', 0) +
        lines[2:-2] +
        [''] * node.get('empty-lines-after', 0)
    )
    prompt = node.get('latex_prompt')
    color = 'nbsphinxin' if prompt.startswith('In') else 'nbsphinxout'
    prefix = r'\textcolor{' + color + '}{' + prompt + '}' if prompt else ''
    for line in code_lines[:1]:
        out.append(prefix + line)
    prefix = ' ' * len(prompt)
    for line in code_lines[1:]:
        out.append(prefix + line)
    if lines[-2].startswith(r'\end{sphinxVerbatim}'):  # Sphinx >= 1.5
        out.append(lines[-2].replace('sphinxVerbatim', 'Verbatim'))
    elif lines[-2].startswith(r'\end{Verbatim}'):  # Sphinx < 1.5
        out.append(lines[-2].replace('Verbatim', 'OriginalVerbatim'))
    else:
        assert False
    assert lines[-1] == ''
    out.append(lines[-1])
    self.body[-1] = '\n'.join(out)


def visit_admonition_html(self, node):
    self.body.append(self.starttag(node, 'div'))
    self.set_first_last(node)
    if self.settings.env.config.html_theme in ('sphinx_rtd_theme', 'julia'):
        if node.children:
            classes = node.children[0]['classes']
            if 'last' not in classes:
                classes.extend(['fa', 'fa-exclamation-circle'])


def depart_admonition_html(self, node):
    self.body.append('</div>\n')


def visit_admonition_latex(self, node):
    # See http://tex.stackexchange.com/q/305898/13684:
    self.body.append('\n\\begin{notice}{' + node['classes'][1] + '}{}\\unskip')


def depart_admonition_latex(self, node):
    self.body.append('\\end{notice}\n')


def do_nothing(self, node):
    pass


def _add_notebook_parser(app):
    """Ugly hack to modify source_suffix and source_parsers.

    Once https://github.com/sphinx-doc/sphinx/pull/2209 is merged (and
    some additional time has passed), this should be replaced by ::

        app.add_source_parser('.ipynb', NotebookParser)

    See also https://github.com/sphinx-doc/sphinx/issues/2162.

    """
    source_suffix = app.config._raw_config.get('source_suffix', ['.rst'])
    if isinstance(source_suffix, sphinx.config.string_types):
        source_suffix = [source_suffix]
    if '.ipynb' not in source_suffix:
        source_suffix.append('.ipynb')
        app.config._raw_config['source_suffix'] = source_suffix
    source_parsers = app.config._raw_config.get('source_parsers', {})
    if '.ipynb' not in source_parsers and 'ipynb' not in source_parsers:
        source_parsers['.ipynb'] = NotebookParser
        app.config._raw_config['source_parsers'] = source_parsers


def setup(app):
    """Initialize Sphinx extension."""
    _add_notebook_parser(app)

    app.add_config_value('nbsphinx_execute', 'auto', rebuild='env')
    app.add_config_value('nbsphinx_execute_arguments', [], rebuild='env')
    app.add_config_value('nbsphinx_allow_errors', False, rebuild='')
    app.add_config_value('nbsphinx_timeout', 30, rebuild='')
    app.add_config_value('nbsphinx_codecell_lexer', 'none', rebuild='env')
    # Default value is set in builder_inited():
    app.add_config_value('nbsphinx_prompt_width', None, rebuild='html')

    app.add_directive('nbinput', NbInput)
    app.add_directive('nboutput', NbOutput)
    app.add_directive('nbinfo', NbInfo)
    app.add_directive('nbwarning', NbWarning)
    app.add_node(CodeNode,
                 html=(do_nothing, depart_code_html),
                 latex=(visit_code_latex, depart_code_latex))
    app.add_node(AdmonitionNode,
                 html=(visit_admonition_html, depart_admonition_html),
                 latex=(visit_admonition_latex, depart_admonition_latex))
    app.connect('builder-inited', builder_inited)
    app.connect('html-page-context', html_page_context)
    app.connect('html-collect-pages', html_collect_pages)
    app.connect('env-purge-doc', env_purge_doc)

    # Make docutils' "code" directive (generated by markdown2rst/pandoc)
    # behave like Sphinx's "code-block",
    # see https://github.com/sphinx-doc/sphinx/issues/2155:
    rst.directives.register_directive('code', sphinx.directives.code.CodeBlock)

    return {'version': __version__, 'parallel_read_safe': True}
