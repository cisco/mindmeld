class CodeGenerator:
    """
    This class contains primitive python functions to generate
    python code
    """

    def begin(self, tab="\t"):
        self.code = []
        self.tab = tab
        self.level = 0

    def end(self):
        return "".join(self.code)

    def write(self, string, new_lines=1):
        self.code.append(self.tab * self.level + string + "\n" * new_lines)

    def indent(self):
        self.level = self.level + 1

    def dedent(self):
        if self.level == 0:
            raise SyntaxError("internal error in code generator")
        self.level = self.level - 1


class MindmeldCodeGenerator(CodeGenerator):
    """
    This class generates MindMeld-specific python code blocks
    """

    def generate_handle(self, params):
        self.write("@app.handle(%s)" % params)

    def generate_header(self, function_name):
        self.write("def %s(request, responder):" % function_name)

    def generate_function(self, handle, function_name, replies):
        self.generate_handle(handle)
        self.generate_header(function_name)
        self.indent()
        self.write("replies = %s" % replies)
        self.write("responder.reply(replies)")
        self.dedent()
        self.write("", 2)

    def generate_top_block(self):
        self.write("# -*- coding: utf-8 -*-")
        self.write('"""This module contains the MindMeld application"""', 2)
        self.write("import copy")
        self.write("from mindmeld import Application")
        self.write("app = Application(__name__)")
        self.write("__all__ = ['app']", 2)

    def generate_follow_up(self, intent, entity, role, replies):
        if role:
            self.write(
                "if not entity_and_roles.get('%s', {}).get('%s'):" % (entity, role)
            )
        else:
            self.write("if not entity_and_roles.get('%s', {}).get(None):" % entity)
        self.indent()
        self.write("replies = %s" % replies)
        self.write(
            "responder.params.allowed_intents = ('unrelated.*', 'app_specific.%s',)"
            % intent
        )
        self.write("responder.reply(replies)")
        self.write("responder.listen()")
        self.write("return")
        self.dedent()
        self.write("")

    def generate_followup_function_code_block(
        self, handle, function_name, intent_entity_role_replies, final_templates
    ):
        self.generate_handle(handle)
        self.generate_header(function_name)
        self.indent()
        self.write("entity_and_roles = copy.deepcopy(dict(request.frame))")
        self.write("for entity in request.entities:")
        self.indent()
        self.write("type = entity['type']")
        self.write("role = entity['role']")
        self.write("val = entity['text']")
        self.write("if type not in entity_and_roles:")
        self.indent()
        self.write("entity_and_roles[type] = {role: val}")
        self.write("responder.frame[type] = {role: val}")
        self.dedent()
        self.write("else:")
        self.indent()
        self.write("entity_and_roles[type].update({role: val})")
        self.write("responder.frame[type].update({role: val})")
        self.dedent()
        self.dedent()
        self.write("")

        for intent in intent_entity_role_replies:
            for entity in intent_entity_role_replies[intent]:
                roles = list(intent_entity_role_replies[intent][entity].keys())
                if len(roles) == 1:
                    replies = intent_entity_role_replies[intent][entity][roles[0]]
                    self.generate_follow_up(intent, entity, None, replies)
                else:
                    for role in roles:
                        replies = intent_entity_role_replies[intent][entity][role]
                        self.generate_follow_up(intent, entity, role, replies)

        self.write("kwargs = {}")
        for intent in intent_entity_role_replies:
            for entity in intent_entity_role_replies[intent]:
                roles = list(intent_entity_role_replies[intent][entity].keys())
                if len(roles) == 1:
                    self.write(
                        "%s_slot = responder.frame['%s'].pop(None)" % (roles[0], entity)
                    )
                    self.write("kwargs['%s'] = %s" % (roles[0], roles[0] + "_slot"))
                else:
                    for role in roles:
                        self.write(
                            "%s_slot = responder.frame['%s'].pop('%s')"
                            % (role, entity, role)
                        )
                        self.write("kwargs['%s'] = %s" % (role, role + "_slot"))

        code_block = "replies = ["
        for template in final_templates:
            code_block += '"' + template + '"' + ".format(**kwargs), "
        code_block += "]"
        self.write(code_block)
        self.write("responder.reply(replies)")
        self.write("return")
        self.dedent()
        self.write("")
