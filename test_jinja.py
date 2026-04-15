from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'))
try:
    template = env.get_template('results.html')
    print("Template parsed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
