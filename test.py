origin_code = '''def render_body(context, options)
      if options.key?(:partial)
        [render_partial(context, options)]
      else
        StreamingTemplateRenderer.new(@lookup_context).render(context, options)
      end
    end
'''
code = []
oc = origin_code
code_new = []
code_new.append('<extra_id_0>')
for i in code:
    code_new.append(i)
    a = oc.find(i)
    if oc[a + len(i):a + len(i) + 1] == '\n':
        code_new.append('<extra_id_1>')
        code_new.append('<extra_id_0>')
    oc = oc[a + len(i):]
code_new.append('<extra_id_1>')
code_new = ' '.join(code_new)

