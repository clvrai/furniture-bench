from docutils import nodes
from docutils.parsers.rst import Directive

class CheckboxDirective(Directive):
    has_content = False
    required_arguments = 1

    def run(self):
        checkbox_id = self.arguments[0]
        wrapper_node = nodes.raw('', '<div class="check_wrap">', format='html')
        input_node = nodes.raw('', f'<input type="checkbox" id="{checkbox_id}"/>', format='html')
        label_node = nodes.raw('', f'<label for="{checkbox_id}"><span></span></label>', format='html')
        wrapper_node += input_node
        wrapper_node += label_node
        return [wrapper_node]

def setup(app):
    app.add_directive("checkbox", CheckboxDirective)
