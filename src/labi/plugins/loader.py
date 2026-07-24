class PluginLoader:
    def __init__(self):
        self.plugins = {}
    def load(self, name, plugin):
        self.plugins[name] = plugin
